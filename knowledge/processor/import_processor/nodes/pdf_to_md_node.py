# 内置包>三方包>自己的包
import json
import subprocess
import time
import zipfile
from os import mkdir
from typing import Tuple
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, T, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, PdfConversionError


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        import_file_path_obj, file_dir_obj = self._validate_state(state)

        processed_code = self._execute_mineru_parse(import_file_path_obj, file_dir_obj)
        if processed_code != 0:
            raise PdfConversionError(node_name=self.name,
                                     message="MinerU解析失败")

        unzip_dir = self._extract_zip(import_file_path_obj, file_dir_obj)

        md_path = self.get_md_path(import_file_path_obj, unzip_dir)

        # 把解析得到的md路径回写到state，供下游md_to_img_node使用
        state["md_path"] = md_path

        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:

        self.log_step("step1", "准备校验和获取解析文件路径和输出目录")

        import_file_path = state.get("import_file_path", "")

        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path")

        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_path",
                                  message="解析文件的路径不存在")

        file_dir = state.get("file_dir", "")

        if not file_dir:
            file_dir = import_file_path_obj.parent

        file_dir_obj = Path(file_dir)

        if not file_dir_obj.exists():
            # file_dir_obj.mkdir(parents=True, exist_ok=True)
            raise StateFieldError(node_name=self.name, field_name="file_dir",
                                  message="输出目录不存在")

        self.logger.info(f"解析的文件路径:{import_file_path}")
        self.logger.info(f"输出的文件目录:{file_dir}")

        return import_file_path_obj, file_dir_obj

    def _execute_mineru_parse(self, import_file_path_obj: Path, file_dir_obj: Path) -> int:
        # self.log_step("step2", "执行解析")
        file_name = import_file_path_obj.stem
        cmd = [
            "mineru-kit",
            "parse",
            str(import_file_path_obj),
            "--format",
            "zip",
            "-o",
            str(file_dir_obj / f"{file_name}.zip"),
            "--tier",
            "standard"
        ]

        start_time = time.time()
        proc = subprocess.Popen(args=cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                errors="replace",
                                encoding="utf-8",
                                bufsize=1)

        for line in proc.stdout:
            self.logger.info(f"MinerU解析产生的日志：{line}")

        processed_result = proc.wait()
        end_time = time.time()
        if processed_result == 0:
            self.logger.info(f"MinerU解析成功,耗时：{end_time - start_time:.2f}s")
        else:
            self.logger.info("MinerU解析失败")

        return processed_result

    def get_md_path(self, import_file_path_obj: Path, unzip_dir: Path) -> str:
        for md in unzip_dir.rglob("*.md"):
            return str(md)
        raise PdfConversionError(node_name=self.name,
                                 message=f"解压产物中未找到md:{unzip_dir}")

    def _extract_zip(self, import_file_path_obj: Path, file_dir_obj: Path):
        zip_path = file_dir_obj / f"{import_file_path_obj.stem}.zip"
        unzip_dir = file_dir_obj / import_file_path_obj.stem

        if not zip_path.exists():
            raise PdfConversionError(node_name=self.name,
                                     message=f"未找到解析产物zip:{zip_path}")

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(unzip_dir)

        zip_path.unlink(missing_ok=True)

        self.logger.info(f"解压产物目录:{unzip_dir}")
        self.logger.info(f"包内成员:{zf.namelist() if False else sorted(p.name for p in unzip_dir.rglob('*'))}")
        self.logger.info(f"已删除压缩包:{zip_path}")

        return unzip_dir

if __name__=="__main__":

    setup_logging()
    pdf_to_md_node = PdfToMdNode()

    init_state = {
        "import_file_path":r"D:\flaskkk111\shopkeeper_brain\knowledge\test\万用表RS-12的使用.pdf",
        "file_dir":r"D:\flaskkk111\shopkeeper_brain\knowledge\test"
    }


    result = pdf_to_md_node.process(init_state)

    result_str = json.dumps(result,indent=4,ensure_ascii=False)
    print(result_str)