from management import (
    UserRequest,
    AIAgent,
    AgentManagementLayer,
    ActionTask,
)


def create_mock_manager():
    agents = [
        AIAgent(
            agent_id="AI Agent #1",
            profile=["action1", "action2", "action3", "action4", "action5"],
        ),
        AIAgent(
            agent_id="AI Agent #2",
            profile=["action4", "action5", "action6"],
        ),
        AIAgent(
            agent_id="AI Agent #3",
            profile=["action8", "action9", "action10"],
        ),
    ]

    return AgentManagementLayer(agents)


def test_preprocess():
    manager = create_mock_manager()

    raw_input = "   Please   Collect   DATA   "
    output = manager.preprocess(raw_input)

    assert output == "please collect data"

    print("test_preprocess passed")


def test_detect_intent_data_collection():
    manager = create_mock_manager()

    text = "please collect dataset from simulation"
    output = manager.detect_intent(text)

    assert output == "data_collection"

    print("test_detect_intent_data_collection passed")


def test_detect_intent_simulation():
    manager = create_mock_manager()

    text = "run simulation scenario"
    output = manager.detect_intent(text)

    assert output == "simulation"

    print("test_detect_intent_simulation passed")


def test_detect_intent_report_generation():
    manager = create_mock_manager()

    text = "generate final report summary"
    output = manager.detect_intent(text)

    assert output == "report_generation"

    print("test_detect_intent_report_generation passed")


def test_detect_intent_general_task():
    manager = create_mock_manager()

    text = "hello agent"
    output = manager.detect_intent(text)

    assert output == "general_task"

    print("test_detect_intent_general_task passed")


def test_plan_actions_data_collection():
    manager = create_mock_manager()

    output = manager.plan_actions(
        intent="data_collection",
        user_input="please collect data",
        location_id="Location #1",
    )

    action_names = [task.action_name for task in output]

    assert len(output) == 5
    assert action_names == [
        "action1",
        "action2",
        "action3",
        "action4",
        "action5",
    ]

    assert output[0].payload["user_input"] == "please collect data"
    assert output[0].payload["location_id"] == "Location #1"

    print("test_plan_actions_data_collection passed")


def test_plan_actions_simulation():
    manager = create_mock_manager()

    output = manager.plan_actions(
        intent="simulation",
        user_input="run simulation scenario",
        location_id="Location #1",
    )

    action_names = [task.action_name for task in output]

    assert action_names == [
        "action4",
        "action5",
        "action6",
    ]

    print("test_plan_actions_simulation passed")


def test_plan_actions_report_generation():
    manager = create_mock_manager()

    output = manager.plan_actions(
        intent="report_generation",
        user_input="generate report",
        location_id="Location #1",
    )

    action_names = [task.action_name for task in output]

    assert action_names == [
        "action8",
        "action9",
        "action10",
    ]

    print("test_plan_actions_report_generation passed")


def test_assign_tasks():
    manager = create_mock_manager()

    tasks = [
        ActionTask(
            task_id="task_001",
            action_name="action1",
            description="Test action 1",
            required_profile="action1",
            payload={},
        ),
        ActionTask(
            task_id="task_002",
            action_name="action8",
            description="Test action 8",
            required_profile="action8",
            payload={},
        ),
    ]

    output = manager.assign_tasks(tasks)

    assigned_agent_ids = {
        agent.agent_id: [task.action_name for task in task_list]
        for agent, task_list in output.items()
    }

    assert assigned_agent_ids["AI Agent #1"] == ["action1"]
    assert assigned_agent_ids["AI Agent #3"] == ["action8"]

    print("test_assign_tasks passed")


def test_execute_tasks():
    manager = create_mock_manager()

    task = ActionTask(
        task_id="task_001",
        action_name="action1",
        description="Understand user request",
        required_profile="action1",
        payload={"user_input": "hello"},
    )

    assigned_tasks = {
        manager.agents[0]: [task],
    }

    output = manager.execute_tasks(assigned_tasks)

    assert len(output) == 1
    assert output[0]["task_id"] == "task_001"
    assert output[0]["agent_id"] == "AI Agent #1"
    assert output[0]["status"] == "success"
    assert output[0]["response"]["action"] == "action1"

    print("test_execute_tasks passed")


def test_aggregate_results():
    manager = create_mock_manager()

    results = [
        {
            "task_id": "task_001",
            "agent_id": "AI Agent #1",
            "status": "success",
            "response": {
                "action": "action1",
                "output": "Completed action1",
            },
        },
        {
            "task_id": "task_002",
            "agent_id": "AI Agent #2",
            "status": "success",
            "response": {
                "action": "action4",
                "output": "Completed action4",
            },
        },
    ]

    output = manager.aggregate_results(results)

    assert output["status"] == "completed"
    assert output["summary"] == "2 actions completed successfully."
    assert len(output["details"]) == 2

    print("test_aggregate_results passed")


def test_handle_request_full_flow():
    manager = create_mock_manager()

    request = UserRequest(
        user_id="user_001",
        raw_input="Please collect data by running simulation scenarios",
        input_type="chat",
        location_id="Location #1",
    )

    output = manager.handle_request(request)

    assert "session_id" in output
    assert output["intent"] == "data_collection"
    assert output["actions"] == [
        "action1",
        "action2",
        "action3",
        "action4",
        "action5",
    ]
    assert output["final_response"]["status"] == "completed"
    assert len(manager.memory) == 1

    print("test_handle_request_full_flow passed")


def run_all_tests():
    test_preprocess()
    test_detect_intent_data_collection()
    test_detect_intent_simulation()
    test_detect_intent_report_generation()
    test_detect_intent_general_task()
    test_plan_actions_data_collection()
    test_plan_actions_simulation()
    test_plan_actions_report_generation()
    test_assign_tasks()
    test_execute_tasks()
    test_aggregate_results()
    test_handle_request_full_flow()

    print("\nAll mock tests passed")


if __name__ == "__main__":
    run_all_tests()