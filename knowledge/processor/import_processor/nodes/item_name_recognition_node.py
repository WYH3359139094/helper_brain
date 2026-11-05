from typing import List, Tuple, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE

class ItemNameRecognitionNode(BaseNode):

    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        file_title, chunks, item_name_chunk_k=self._validate_state(state)

        item_name_context = self._prepare_llm_context(chunks, item_name_chunk_k)

        item_name = self._recognition_item_name(item_name_context, file_title)

        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        self._insert_milvus(dense_vector, sparse_vector, file_title, item_name)

        self._fill_item_name(state, chunks, item_name)

        return state

    def _validate_state(self, state)-> Tuple[str, List, int]:

        file_title = state.get("file_title")

        if not file_title:
            raise StateFieldError(node_name=self.name,field_name="file_title",expected_type=str)

        chunks = state.get("chunks")

        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name,field_name="chunks",expected_type=list)

        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size

        if not item_name_chunk_k or item_name_chunk_k <=0:
            raise ValidationError(message="商品名识别的辅助切片不合法")

        if not item_name_chunk_size or item_name_chunk_size <=0:
            raise ValidationError(message="商品名识别的辅助切片长度不合法")

        return  file_title, chunks, item_name_chunk_k

    def _prepare_llm_context(self, chunks:List[Dict], item_name_chunk_k:int) ->str:

        # 1. 遍历
        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):

            # 1.1 不是字典类型 直接过滤掉该块
            if not isinstance(chunk, dict):
                continue

            # 1.2 获取chunk的content
            content = chunk.get('content')

            splice_context = f"【切片】- f{index}- {content}"

            final_context.append(splice_context)

        return "\n".join(final_context)

    def _recognition_item_name(self, item_name_context: str, file_title: str) -> str:

        # 1.调用LLM客户端
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"OpenAI 的LLM客户端创建失败,降级使用文件标题{file_title}作为商品名 {str(e)}")
            return file_title

        # 2. 调用LLM模型
        # 2.1 获取商品名识别的llm系统提示词
        system_prompt = ITEM_NAME_SYSTEM_PROMPT

        # 2.2 获取商品名识别的llm用户提示词
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)

        try:
            # 3. 调用 返回AIMessage对象
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 4. 获取AI回复的具体内容
            llm_result = llm_response.content.strip('')
            if not llm_result or llm_result == 'UNKNOWN':
                self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名兜底")
                return file_title

            self.logger.info(f"LLM为文档：{file_title} 提取的商品名：{llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名: {str(e)}")
            return file_title

    def _embedding_item_name(self, item_name: str) -> Tuple[Optional[List], Optional[Dict[str, Any]]]:
        try:
            bge_m3_client: BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE_M3嵌入模型客户端创建失败: {str(e)}")
            return None, None
        try:
            # 2. 计算稠密和稀疏向量
            vector_result = bge_m3_client.encode_documents(documents=[item_name])

            # 3. 解析稠密向量和稀疏向量
            # 3.1 获取稠密向量
            dense_vector = vector_result.get('dense')[0].tolist()
            # 3.2 获取稀疏向量矩阵csr(解开)
            sparse_csr = vector_result.get('sparse')
            # 3.1 获取行索引
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            # 3.2 获取token_id
            token_id = sparse_csr.indices[start_index:end_index].tolist()
            # 3.3 获取weight
            weight = sparse_csr.data[start_index:end_index].tolist()
            # 3.4 构建{"token_id":weight}字典结构

            sparse_vector = dict(zip(token_id, weight))
            self.logger.info(f"计算出来的稠密向量的维度: {len(dense_vector)}")
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"BGE-M3嵌入模型计算{item_name}向量失败 原因：{str(e)}")
            return None, None

    def _insert_milvus(self, dense_vector: List, sparse_vector: Dict[str, Any], file_title: str, item_name: str):


        # 1. 判断稠密向量和稀疏向量是否都存在
        if not dense_vector or not sparse_vector:
            return

        # 2. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端创建失败,原因：{str(e)}")
            return

        # 3. Milvus三大核心概念（集合：Collection[1.集合名 2.字段约束：schema 3.索引]）

        # 4. 获取集合名字
        item_name_collection_name = self.config.item_name_collection
        try:
            # 5. 创建Milvus集合(幂等性校验)
            if not milvus_client.has_collection(item_name_collection_name):
                self._create_item_name_collection(item_name_collection_name, milvus_client)

            # 6. 构建数据行
            item_name_data_row = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            # 7. 插入数据
            inserted_result = milvus_client.insert(collection_name=item_name_collection_name, data=[item_name_data_row])
        except Exception as e:
            self.logger.error(f"商品名{item_name}插入失败 {str(e)}")

        self.logger.info(f"插入的结果:{inserted_result},主键值:{inserted_result.get('ids')}")

    def _create_item_name_collection(self, item_name_collection_name: str, milvus_client: MilvusClient):

        # 1. 创建schema约束
        schema = milvus_client.create_schema()
        # 1.1 创建主键字段约束
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=10)

        # 1.2 创建标量字段的约束
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)

        # 1.3 创建向量字段的约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引(标量字段建立索引 向量字段建立索引)
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"  # IP COSINE L2
            # Milvus计算出来的稠密向量已经进行了归一化处理 所以度量类型选择COSINE或者IP效果一样，但是如果使用别的方式计算出来的稠密向量没有经过归一化处理 那么COSINE和IP就不相等
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP"  # 只有IP 和BM25
        )

        # 3. 创建集合
        milvus_client.create_collection(collection_name=item_name_collection_name,
                                        schema=schema, index_params=index_params)

        self.logger.info(f"创建{item_name_collection_name}集合成功")

    def _fill_item_name(self, state: ImportGraphState, chunks: List[Dict], item_name: str):

        # 1. 更新chunk的item_name
        for chunk in chunks:
            chunk['item_name'] = item_name

        # 2. 更新state中的item_name
        state['item_name'] = item_name
