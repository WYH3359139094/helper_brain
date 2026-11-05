"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging, time

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.exceptions import ImportProcessError
from knowledge.utils.task_util import add_running_task, add_done_task, add_node_duration

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:

        task_id = state.get('task_id', '')
        try:
            # 1. 开始准备执行节点
            self.logger.info(f"--- {self.name} 开始 ---")

            # 2. 执行节点
            if task_id:
                add_running_task(task_id, self.name)
            start_time = time.time()
            result = self.process(state)
            end_time = time.time()
            if task_id:
                add_done_task(task_id, self.name)
                add_node_duration(task_id, self.name, end_time-start_time)

            # 3. 执行节点成功
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:

        pass

    def log_step(self, step_name: str, message: str = ""):

        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# 配置日志格式
def setup_logging(level: int = logging.INFO):

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
