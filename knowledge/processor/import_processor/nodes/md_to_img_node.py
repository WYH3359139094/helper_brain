import re
import base64
from collections import deque
from pathlib import Path
from dataclasses import dataclass
import time


from logging import Logger
from typing import Tuple, List, Set, Optional, Dict, Deque
from openai import OpenAI

from knowledge.processor.import_processor.exceptions import StateFieldError, FileProcessingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients

@dataclass
class ImageContext:

    head:str
    pre_text:str
    post_text:str


@dataclass
class ImageInfo:

    name:str
    path:str
    imag_context:ImageContext


class _VLMSummarizer:

    def __init__(self, logger: Logger, requests_per_minute: int):
        self.logger = logger
        self.requests_per_minute = requests_per_minute

    def _summary_all(self, document_name:str, img_info_list:List[ImageInfo], vl_model:str)->Dict[str,str]:

        summaries = {}
        request_timestamps:Deque[float] = deque()
        try:
            vlm_client=AIClients.get_vlm_client()

        except Exception as e:
            for img_info in img_info_list:
                summaries[img_info.name]="暂无摘要"
            return summaries

        for img_info in img_info_list:
            self._enforce_rate_limit(request_timestamps, self.requests_per_minute)
            summaries[img_info.name] =self._summary_one(document_name,img_info,vlm_client,vl_model)

        self.logger.info(f"生产{len(summaries)}个图片摘要")
        return summaries

    def _summary_one(self, document_name:str, img_info:ImageInfo, vlm_client:OpenAI, vl_model:str):
        # 1. 构造VLM需要的上下文（标题名、上文内容、下文内容）
        parts = [p for p in
                 (img_info.imag_context.head, img_info.imag_context.pre_text, img_info.imag_context.post_text) if p]

        # 2. 构建最终的上下文
        final_context = "\n".join(parts) if parts else "暂无上下文"

        # 3. 根据图片地址获取到图片的内容（二进制字节流）---文本协议认识（base64编码）--->解码（‘utf-8’）--->字符串（文本协议能传输） ---- 根据收到字符串解码（二进制字节流 还原图片内容）
        try:
            with  open(img_info.path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
        except IOError as e:
            self.logger.error(f"读取图片文件{img_info.path} 内容失败: {e}")
            return "暂无图片描述"

        # 4. 利用vlm客户端调用VLM模型
        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"

    def _enforce_rate_limit(
            self, timestamps: Deque[float],
            max_requests: int,
            window: int = 60,
    ):
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)

class _ImageUploader:

    def __init__(self, logger:Logger):
        self.logger = logger

    def upload_and_replace(self,object_dir_name:str ,md_content:str, img_info_list:List[ImageInfo], summaries:Dict[str,str], minio_url:str, minio_bucket_name:str):
        # 1. 上传
        remote_urls = self._upload_all(object_dir_name, img_info_list, minio_url, minio_bucket_name)

        # 2. 更新
        md_content = self._update_md(md_content, summaries, remote_urls)

        return md_content

    def _upload_all(self, object_dir_name: str, img_info_list: List[ImageInfo], minio_url: str,
                    minio_bucket_name: str) -> Dict[str, str]:

        remote_urls = {}
        # 1. 得到MinIO客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            for img_info in img_info_list:
                remote_urls[img_info.name] = img_info.path
            return remote_urls

        # 2. 遍历上传每一个
        for img_info in img_info_list:
            object_name = f"{object_dir_name}/{img_info.name}"
            try:
                # 2.1 上传图片到MinIO
                minio_client.fput_object(
                    minio_bucket_name, object_name, img_info.path)
                # 2.2 自己拼装路径
                # http://192.168.200.145:9000/桶名/对象名
                self.logger.info(f"成功将图片{img_info.name}上传到MinIO中")
                remote_urls[img_info.name] = f"{minio_url}/{minio_bucket_name}/{object_name}"
            except Exception as e:
                self.logger.warn(f"上传图片{img_info.name}到MinIO失败，用本地图片地址做兜底")
                remote_urls[img_info.name] = img_info.path

        self.logger.info(f"获取到远程的{len(remote_urls)}图片地址")
        return remote_urls

    def _update_md(self, md_content: str, summaries: Dict[str, str], remote_urls: Dict[str, str]) -> str:
        """
        更新MD中的图片描述和远程图片地址
        Args:
            md_content:  md内容
            summaries:   vlm生成的摘要
            remote_urls: minio生成的url

        Returns:
            新md

        """
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            """

            Args:
                match:

            Returns:
                ![摘要](远程图片地址)
            """

            for img_name, img_summary in summaries.items():
                origin_img_path = match.group(2)
                img_name_in_md = Path(origin_img_path).name
                if img_name == img_name_in_md:
                    return f"![{img_summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)


class _MdFileHandler:

    def __init__(self, logger:Logger,node_name:str):
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self,state)->Tuple[str,Path,Path]:
        md_path = state.get("md_path","")

        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name="md_path",expected_type=str)

        md_path_obj = Path(md_path)

        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=Path)
        try:
            with open(md_path_obj, 'r', encoding='utf-8') as f:
                md_content = f.read()

        except IOError as e:
            self.logger.error(f"MD文件: {md_path_obj} 打开失败")
            raise FileProcessingError(message="文件打开失败",node_name=self.node_name)

        img_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, img_dir

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise FileProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)




class _ImageScanner:

    def __init__(self, logger: Logger):
        self.logger = logger

    def scan_imgs_dir(self,img_dir_obj:Path,md_content:str,image_extensions:Set[str],img_content_length:int)->List[ImageInfo]:

        img_info_list = []

        for img_path in img_dir_obj.iterdir():
            if not img_path.is_file():
                self.logger.error(f"{img_path}不是一个有效的文件")
                continue

            if not img_path.suffix in image_extensions:
                self.logger.error(f"{img_path}不是允许的图片后缀格式")
                continue

            ctx = self._find_context(img_path.name,md_content,img_content_length)

            if not ctx:
                self.logger.info(f"MD中未找到该图片{img_path.name}引用")
                continue

            img_info_list.append(ImageInfo(
                name=img_path.name,
                path=str(img_path),
                imag_context=ctx
            ))

        self.logger.info((f"MD中找到{len(img_info_list)}个有效的图片"))

        return img_info_list

    def _find_context(self, img_name:str, md_content:str, img_content_length:int)->Optional[ImageContext]:
        pattern=re.compile(r"!\[.*?\]\(.*?"+re.escape(img_name)+r".*?\)")

        md_lines = md_content.split("\n")

        for md_idx,md_line in enumerate(md_lines):

            if not pattern.search(md_line):
                continue

            head,prev_index  =self._find_heading_up(md_lines,md_idx)
            pre_lines=md_lines[prev_index+1:md_idx]
            pre_context = self._extract_limited_context(pre_lines, img_content_length,direction="front")

            next_index = self._find_heading_down(md_lines,md_idx)
            next_lines = md_lines[md_idx+1:next_index]
            post_context = self._extract_limited_context(next_lines, img_content_length,direction="back")

            return ImageContext(
                head = head,
                pre_text=pre_context,
                post_text=post_context
            )
        return None


    def _find_heading_up(self, md_lines:List[str], from_idx:int)->Tuple[str,int]:

        for i in range(from_idx-1, -1, -1):
            if re.match(r"#{1,6}\s+",md_lines[i]):
                return md_lines[i],i

        return "",-1

    def _find_heading_down(self, md_lines:List[str], from_idx:int)->int:
        for i in range(from_idx+1,len(md_lines)):
            if re.match(r"#{1,6}\s+", md_lines[i]):
                return i

        return len(md_lines)

    def _extract_limited_context(self, extracted_md_lines:List[str], img_content_length:int, direction:str)->str:

        current_paragraph = []
        paragraphs= []

        for line in extracted_md_lines:

            is_blank_line = not line.strip()

            is_other_image = re.match(
                r"!\[.*?\]\(.*?\)$", line.strip()
            )

            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph=[]
                continue

            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        if direction=="front":
            paragraphs.reverse()

        total = 0
        selected=[]

        for paragraph in paragraphs:
            if total+len(paragraph)>img_content_length and selected:
                break

            selected.append(paragraph)
            total+=len(paragraph)
        if direction=="front":
            selected.reverse()
        return "\n\n".join(selected)

class MarkDownToImgNode(BaseNode):

    def __init__(self):
        super().__init__()
        self._md_file_handler = _MdFileHandler(self.logger,self.name)
        self._img_scaner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger,self.config.requests_per_minute)
        self._img_uploader = _ImageUploader(self.logger)


    name = "md_to_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        config = self.config

        self.log_step("step1","读取MD内容，路径以及图片的目录")

        md_content,md_path_obj,img_dir_obj = self._md_file_handler.validate_and_read_md(state)
        if not img_dir_obj.exists():
            state["md_content"]=md_content
            return state

        self.log_step("step2","准备开始扫描图片目录")
        img_info_list:List[ImageInfo] = self._img_scaner.scan_imgs_dir(img_dir_obj,
                                                                       md_content,
                                                                       config.image_extensions,
                                                                       config.img_content_length
                                                                       )

        self.log_step("step3", "利用VLM提取摘要")
        summaries:Dict[str,str]=self._vlm_summarizer._summary_all(md_path_obj.stem,img_info_list,config.vl_model)

        self.log_step("step4", "上传文件到MinIO,且更新MD")
        new_md_content = self._img_uploader.upload_and_replace(md_path_obj.stem, md_content, img_info_list, summaries, config.get_minio_base_url(), config.minio_bucket)

        self._md_file_handler.backup(md_path_obj, new_md_content)

        state["md_content"]=new_md_content

        return state



if __name__=="__main__":
    setup_logging()
    md_img_node = MarkDownToImgNode()
    init_state={
        "md_path":r"D:\flaskkk111\shopkeeper_brain\knowledge\test\万用表RS-12的使用\markdown.md"
    }
    md_img_node.process(init_state)