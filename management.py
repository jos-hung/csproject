from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import uuid
import time

if TYPE_CHECKING:
    from query import QueryAgent


@dataclass
class UserRequest:
    user_id: str
    raw_input: str
    input_type: str  # "voice" | "chat"
    location_id: str


@dataclass
class ActionTask:
    task_id: str
    action_name: str
    description: str
    required_profile: str
    payload: Dict[str, Any]


@dataclass
class AgentResult:
    task_id: str
    agent_id: str
    status: str
    response: Dict[str, Any]


class AIAgent:
    def __init__(self, agent_id: str, profile: List[str]):
        self.agent_id = agent_id
        self.profile = profile

    def can_handle(self, action_name: str) -> bool:
        return action_name in self.profile

    def execute(self, task: ActionTask) -> AgentResult:
        print(f"[{self.agent_id}] executing {task.action_name}")

        result = {
            "action": task.action_name,
            "description": task.description,
            "output": f"Completed {task.action_name}",
            "payload": task.payload,
        }

        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            status="success",
            response=result,
        )


class AgentManagementLayer:
    """
    Communication Coordination Layer

    Nhiệm vụ:
    1. Nhận request từ AI Avatar / Voice / Chat
    2. Tiền xử lý input
    3. Phân tích intent
    4. Tạo danh sách action
    5. Chỉ định action cho agent phù hợp
    6. Thu thập response
    7. Ghi log / dataset / report
    """

    def __init__(
        self,
        agents: List[AIAgent],
        query_agent: Optional["QueryAgent"] = None,
    ):
        self.agents = agents
        self.memory = []
        self.query_agent = query_agent

    def handle_request(self, request: UserRequest) -> Dict[str, Any]:
        session_id = str(uuid.uuid4())

        preprocessed_input = self.preprocess(request.raw_input)

        intent = self.detect_intent(preprocessed_input)

        action_plan = self.plan_actions(
            intent=intent,
            user_input=preprocessed_input,
            location_id=request.location_id,
        )

        assigned_tasks = self.assign_tasks(action_plan)

        results = self.execute_tasks(assigned_tasks)

        final_response = self.aggregate_results(results)

        self.save_record(
            session_id=session_id,
            request=request,
            intent=intent,
            action_plan=action_plan,
            results=results,
            final_response=final_response,
        )

        return {
            "session_id": session_id,
            "intent": intent,
            "actions": [task.action_name for task in action_plan],
            "agent_results": results,
            "final_response": final_response,
        }

    def preprocess(self, raw_input: str) -> str:
        """
        Tiền xử lý:
        - chuẩn hóa text
        - loại bỏ noise
        - có thể thêm speech-to-text nếu input là voice
        """
        text = raw_input.strip()
        text = text.lower()
        text = " ".join(text.split())
        return text

    def detect_intent(self, text: str) -> str:
        """
        Phân tích intent từ văn bản đầu vào.
        Nếu có QueryAgent, sử dụng LLM để phân loại.
        Ngược lại, dùng keyword-based fallback.
        """
        if self.query_agent is not None:
            return self.query_agent.detect_intent(text)

        if "collect" in text or "record" in text or "dataset" in text:
            return "data_collection"

        if "simulate" in text or "scenario" in text:
            return "simulation"

        if "report" in text or "summary" in text:
            return "report_generation"

        return "general_task"

    def plan_actions(
        self,
        intent: str,
        user_input: str,
        location_id: str,
    ) -> List[ActionTask]:
        """
        Chia request thành các action nhỏ hơn.
        Đây là phần giống 'mastermind'.
        """

        if intent == "data_collection":
            actions = [
                ("action1", "Prepare simulation scenario"),
                ("action2", "Assign roles to virtual agents"),
                ("action3", "Run interaction in virtual environment"),
                ("action4", "Record video, input and response"),
                ("action5", "Store collected data into database"),
            ]

        elif intent == "simulation":
            actions = [
                ("action4", "Load virtual environment"),
                ("action5", "Generate scenario"),
                ("action6", "Coordinate multi-agent interaction"),
            ]

        elif intent == "report_generation":
            actions = [
                ("action8", "Fetch records from database"),
                ("action9", "Analyze collected responses"),
                ("action10", "Generate final report"),
            ]

        else:
            actions = [
                ("action1", "Understand user request"),
                ("action4", "Select appropriate environment"),
                ("action8", "Generate response summary"),
            ]

        return [
            ActionTask(
                task_id=str(uuid.uuid4()),
                action_name=action_name,
                description=description,
                required_profile=action_name,
                payload={
                    "user_input": user_input,
                    "location_id": location_id,
                    "timestamp": time.time(),
                },
            )
            for action_name, description in actions
        ]

    def assign_tasks(self, tasks: List[ActionTask]) -> Dict[AIAgent, List[ActionTask]]:
        """
        Chọn agent phù hợp dựa trên profile action.
        Ví dụ:
        Agent #1: action1-5
        Agent #2: action4-6
        Agent #3: action8-10
        """
        assignment: Dict[AIAgent, List[ActionTask]] = {}

        for task in tasks:
            selected_agent = None

            for agent in self.agents:
                if agent.can_handle(task.action_name):
                    selected_agent = agent
                    break

            if selected_agent is None:
                raise ValueError(f"No agent can handle {task.action_name}")

            assignment.setdefault(selected_agent, []).append(task)

        return assignment

    def execute_tasks(
        self,
        assigned_tasks: Dict[AIAgent, List[ActionTask]],
    ) -> List[Dict[str, Any]]:
        results = []

        for agent, tasks in assigned_tasks.items():
            for task in tasks:
                result = agent.execute(task)
                results.append({
                    "task_id": result.task_id,
                    "agent_id": result.agent_id,
                    "status": result.status,
                    "response": result.response,
                })

        return results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tổng hợp response từ nhiều agent.
        Nếu có QueryAgent, sử dụng LLM để tạo tóm tắt.
        Ngược lại, dùng simple count-based summary.
        """
        if self.query_agent is not None:
            return self.query_agent.aggregate_results(results)

        successful_results = [
            result for result in results
            if result["status"] == "success"
        ]

        return {
            "status": "completed",
            "summary": f"{len(successful_results)} actions completed successfully.",
            "details": successful_results,
        }

    def save_record(
        self,
        session_id: str,
        request: UserRequest,
        intent: str,
        action_plan: List[ActionTask],
        results: List[Dict[str, Any]],
        final_response: Dict[str, Any],
    ):
        """
        Giả lập lưu DB.
        Thực tế có thể lưu MongoDB, PostgreSQL, vector DB, hoặc file dataset.
        """
        record = {
            "session_id": session_id,
            "user_id": request.user_id,
            "raw_input": request.raw_input,
            "input_type": request.input_type,
            "location_id": request.location_id,
            "intent": intent,
            "actions": [
                {
                    "task_id": task.task_id,
                    "action_name": task.action_name,
                    "description": task.description,
                }
                for task in action_plan
            ],
            "results": results,
            "final_response": final_response,
        }

        self.memory.append(record)
        print("[DB] record saved")
