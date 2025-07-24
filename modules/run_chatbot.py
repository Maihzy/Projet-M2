from modules.agent_manager import AgentManager

_chatbot_instance = None

def get_chatbot():
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = AgentManager()
    return _chatbot_instance

def run_chatbot(user_message, dataset=None, dataset_name=None):
    chatbot = get_chatbot()
    if dataset is not None:
        chatbot.current_dataset = dataset
        chatbot.current_dataset_name = dataset_name
    # Ici, on peut choisir scoring_tool ou explanation_tool selon le contexte
    return chatbot.scoring_tool({'message': user_message})

if __name__ == "__main__":
    chatbot = get_chatbot()
    print(chatbot.welcome_message)
    while True:
        user_input = input("Vous : ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chatbot.scoring_tool({'message': user_input})
        print(f"IA : {response}")