# Import necessary libraries

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

from openai import OpenAI

# Set OpenAI API key
# set OPENAI_API_KEY via export in terminal 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# # Initialize the model and tokenizer for the open-source LLM
# model_name = "EleutherAI/gpt-neo-2.7B"  # Adjust to desired LLM; see options below
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Set up FAISS vector database for retrieval-augmented generation
# index = faiss.IndexFlatL2(768)  # Initialize FAISS index for vector storage

# Expanded Checklist Data
checklist_data = [
    # DESIGN - Concept
    ("Design - Concept", "Define purpose & goal of project"),
    ("Design - Concept", "Identify target audience"),
    ("Design - Concept", "Inclusivity: Determine if it’s for all people or a specific audience"),
    ("Design - Concept", "Team roles: Identify person responsible for resolving bias"),

    # DESIGN - Scope
    ("Design - Scope", "Define possible outcomes and relevant criteria"),
    ("Design - Scope", "Transparency: Clarify relationships between inputs and outcomes"),
    ("Design - Scope", "Monitor sensitive characteristics for impact on outputs"),
    ("Design - Scope", "Expected results: Check for unacknowledged bias by consulting a diverse audience"),

    # DESIGN - Data
    ("Design - Data", "Acquire data (source, freshness, technology used)"),
    ("Design - Data", "Evaluate data for human error, consent, and collection context"),
    ("Design - Data", "Assess incentives during data collection"),
    ("Design - Data", "Data diversity: Ensure inclusivity and fair representation"),
    ("Design - Data", "Data integrity: Anonymize/pseudonymize as needed"),
    ("Design - Data", "Constrain: Establish logical relationships among variables"),
    ("Design - Data", "Assumptions: Explicitly state assumptions and check demographic fairness"),

    # DESIGN - Define Success
    ("Design - Define Success", "Error tolerance: Define acceptable error margins"),
    ("Design - Define Success", "System reliability: Ensure transparent performance expectations"),

    # BUILD - Data Processing
    ("Build - Data Processing", "Choose analytic methods, assess transparency and explainability"),
    ("Build - Data Processing", "Review risks related to chosen method"),

    # BUILD - Tools
    ("Build - Tools", "Choose library tools, verify sources for bias and reliability"),
    ("Build - Tools", "Build from scratch if needed, test for fairness and bias"),

    # BUILD - Feedback Mechanism
    ("Build - Feedback Mechanism - Internal", "Track and analyze anomalous results"),
    ("Build - Feedback Mechanism - Internal", "Assign responsibility for addressing technical biases"),
    ("Build - Feedback Mechanism - External", "Allow user feedback on potentially problematic outcomes"),

    # TEST - Re-evaluate Variables and Data
    ("Test - Re-evaluate Variables and Data", "Adjust variables and/or data to address errors"),
    ("Test - Re-evaluate Variables and Data", "Process new data with same inquiry as original model"),

    # TEST - Identify Errors
    ("Test - Identify Errors", "Examine error distribution across demographics"),
    ("Test - Identify Errors", "Check if error type is consistent across groups (false positives/negatives)"),

    # TEST - Evaluate Effect of Error
    ("Test - Evaluate Effect of Error", "Assess individual impact of false positives and negatives"),

    # TEST - Audit Results
    ("Test - Audit Results", "Check model performance against design expectations"),
    ("Test - Audit Results", "Ensure feedback mechanisms track anomalies effectively"),

    # TEST - Run Model
    ("Test - Run Model", "Test model on representative datasets with diverse demographics"),

    # IMPLEMENT - Monitor Results
    ("Implement - Monitor Results", "Critically examine results for disparate impacts"),
    ("Implement - Monitor Results", "Provide user appeal and reporting mechanisms for unfair treatment"),
    ("Implement - Monitor Results", "Ensure data security and confidentiality"),

    # IMPLEMENT - Contextualize Results
    ("Implement - Contextualize Results", "Interpret results and evaluate performance across demographics"),

    # IMPLEMENT - Quality of Results
    ("Implement - Quality of Results", "Assess confidence in results given data limitations"),
    ("Implement - Quality of Results", "Determine if results are a reliable basis for decision-making")
]

# Prepare checklist data as a list of strings
texts = [f"{main_step} - {sub_step}" for main_step, sub_step in checklist_data]

# Initialize OpenAI embeddings and FAISS vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(texts, embeddings)

# Dialogue State Tracker to track progress
class DialogueStateTracker:
    def __init__(self, checklist):
        self.state_index = 0  # Start at the beginning
        self.checklist = checklist

    def update_state(self):
        if self.state_index < len(self.checklist) - 1:
            self.state_index += 1  # Progress to next checklist item

    def get_state(self):
        return self.checklist[self.state_index]

# Instantiate the dialogue state tracker
state_tracker = DialogueStateTracker(checklist_data)

# Retrieval Module to get relevant checklist items
def retrieve_checklist_items(query, k=5):
    results = vector_store.similarity_search(query, k=k)
    return [item.page_content for item in results]

# Updated Generation Module with correct OpenAI API syntax
def generation_module(context, state, best_response):
    prompt_template = PromptTemplate(
        template="""
        Given the context of the conversation so far: {context}
        And your current step in the checklist: {state}

        Based on this information, provide guidance for this step:
        {best_response}
        """,
        input_variables=["context", "state", "best_response"]
    )
    prompt = prompt_template.format(context=context, state=" - ".join(state), best_response=best_response)

    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an ethical design assistant helping users through an AI checklist."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=100,
    temperature=0.7)

    return response.choices[0].message.content

# Function to evaluate and provide targeted follow-up questions
def generate_followup_questions(user_response, checklist_step):
    followup_prompt = f"""
    The user is working through a checklist step: {checklist_step}.
    Their response was: "{user_response}"

    Identify specific gaps in the user’s understanding based on their response, and provide two or three targeted, specific questions that would help guide the user to a deeper understanding of this step.
    """
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an AI that generates targeted questions to help users deepen their understanding."},
        {"role": "user", "content": followup_prompt}
    ],
    max_tokens=500,
    temperature=0.2)
    return response.choices[0].message.content.strip()

# Conversational Agent Function to manage full interaction
def conversation_agent(query, context=""):
    current_state = state_tracker.get_state()
    best_response = retrieve_checklist_items(query, k=1)[0]

    while True:
        # Generate guidance for the current step
        print("Current State:", current_state)
        guidance = generation_module(context, current_state, best_response)
        print("AI Assistant:", guidance)

        # Get user's response
        user_response = input("You: ")

        # Generate follow-up questions based on the user’s response
        followup_questions = generate_followup_questions(user_response, current_state)

        # If no more follow-up questions are generated, the user may proceed to the next step
        if "no more questions" in followup_questions.lower():
            print("AI Assistant: Great! You've demonstrated a good understanding of this step.")
            state_tracker.update_state()  # Move to the next checklist step
            return guidance

        # Present follow-up questions to help the user expand on their answer
        print("AI Assistant: To deepen your understanding, consider these questions:")
        print(followup_questions)

# Main loop for testing the entire conversation
print("Starting the AI Checklist Conversational Agent. Type 'exit' to end the conversation.\n")

context = "User is seeking guidance on a project."

while True:
    # Get user input
    user_query = input("You: ")
    if user_query.lower() == "exit":
        print("Ending the conversation.")
        break

    # Generate response from the agent
    response = conversation_agent(user_query, context)
    print("AI Assistant:", response)

    # Update context to include the latest user input
    context += f" User asked: {user_query}. AI responded: {response}"
