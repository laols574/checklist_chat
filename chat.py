import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import torch

# Set the open-source model
#model_name = "EleutherAI/gpt-neo-125M" #"EleutherAI/gpt-neo-1.3B"  # or "EleutherAI/gpt-neo-125M"
cache_dir = "/var/scratch/lol201/transformers_cache"

tokenizer = AutoTokenizer.from_pretrained(cache_dir)
model = AutoModelForCausalLM.from_pretrained(cache_dir)

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
    ("Design - Define Success", "System reliability: Ensure transparent performance expectations")

#     # BUILD - Data Processing
#     ("Build - Data Processing", "Choose analytic methods, assess transparency and explainability"),
#     ("Build - Data Processing", "Review risks related to chosen method"),

#     # BUILD - Tools
#     ("Build - Tools", "Choose library tools, verify sources for bias and reliability"),
#     ("Build - Tools", "Build from scratch if needed, test for fairness and bias"), 

#    # BUILD - Feedback Mechanism
#     ("Build - Feedback Mechanism - Internal", "Track and analyze anomalous results"),
#     ("Build - Feedback Mechanism - Internal", "Assign responsibility for addressing technical biases"),
#     ("Build - Feedback Mechanism - External", "Allow user feedback on potentially problematic outcomes"),

#     # TEST - Re-evaluate Variables and Data
#     ("Test - Re-evaluate Variables and Data", "Adjust variables and/or data to address errors"),
#     ("Test - Re-evaluate Variables and Data", "Process new data with same inquiry as original model"),

#     # TEST - Identify Errors
#     ("Test - Identify Errors", "Examine error distribution across demographics"),
#     ("Test - Identify Errors", "Check if error type is consistent across groups (false positives/negatives)"),

#     # TEST - Evaluate Effect of Error
#     ("Test - Evaluate Effect of Error", "Assess individual impact of false positives and negatives"),

#     # TEST - Audit Results
#     ("Test - Audit Results", "Check model performance against design expectations"),
#     ("Test - Audit Results", "Ensure feedback mechanisms track anomalies effectively"),

#     # TEST - Run Model
#     ("Test - Run Model", "Test model on representative datasets with diverse demographics"),

#     # IMPLEMENT - Monitor Results
#     ("Implement - Monitor Results", "Critically examine results for disparate impacts"),
#     ("Implement - Monitor Results", "Provide user appeal and reporting mechanisms for unfair treatment"),
#     ("Implement - Monitor Results", "Ensure data security and confidentiality"),

#     # IMPLEMENT - Contextualize Results
#     ("Implement - Contextualize Results", "Interpret results and evaluate performance across demographics"),

#     # IMPLEMENT - Quality of Results
#     ("Implement - Quality of Results", "Assess confidence in results given data limitations"),
#     ("Implement - Quality of Results", "Determine if results are a reliable basis for decision-making")
]

# Ensure model runs on GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# Define embeddings for FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
texts = [f"{main_step} - {sub_step}" for main_step, sub_step in checklist_data]
vector_store = FAISS.from_texts(texts, embeddings)



class DialogueStateTracker:
    def __init__(self, checklist):
        self.state_index = 0  # Start at the beginning
        self.checklist = checklist

    def update_state(self):
        if self.state_index < len(self.checklist) - 1:
            self.state_index += 1  # Progress to next checklist item

    def get_state(self):
        return self.checklist[self.state_index]

state_tracker = DialogueStateTracker(checklist_data)

# Retrieval Module
def retrieve_checklist_items(query, k=5):
    results = vector_store.similarity_search(query, k=k)
    return [item.page_content for item in results]

# Generation Module
def generation_module(context, state, best_response):
    prompt_template = PromptTemplate(
        template="""
        Given the context: {context}
        And your current step in the checklist: {state}

        Provide guidance for this step:
        {best_response}
        """,
	input_variables=["context", "state", "best_response"]
    )
    prompt = prompt_template.format(context=context, state=" - ".join(state), best_response=best_response)

    # Generate response with the LLM
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
    response = generator(prompt, max_length=1024, num_return_sequences=1)
    return response[0]["generated_text"]

# Function to generate follow-up questions
def generate_followup_questions(user_response, checklist_step):
    followup_prompt = f"""
    The user is working through a checklist step: {checklist_step}.
    Their response was: "{user_response}"

    Provide specific follow-up questions to deepen understanding:
    """
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
    response = generator(followup_prompt, max_length=300, num_return_sequences=1)
    return response[0]["generated_text"]


# Conversational Agent
def conversation_agent(query, context=""):
    current_state = state_tracker.get_state()
    best_response = retrieve_checklist_items(query, k=1)[0]

    while True:
        # Generate guidance
        print("Current State:", current_state)
        guidance = generation_module(context, current_state, best_response)
        print("AI Assistant:", guidance)

        # Get user response
        user_response = input("You: ")

        # Generate follow-up questions
        followup_questions = generate_followup_questions(user_response, current_state)
        if "no more questions" in followup_questions.lower():
            print("AI Assistant: Great! You've demonstrated a good understanding of this step.")
            state_tracker.update_state()
            return guidance

        print("AI Assistant: Consider these follow-up questions:")
        print(followup_questions)

# Main Loop
print("Starting the Open-Source LLM Conversational Agent. Type 'exit' to end the conversation.\n")
context = "User is seeking guidance on a project."

while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        print("Ending the conversation.")
        break

    response = conversation_agent(user_query, context)
    print("AI Assistant:", response)
    context += f" User asked: {user_query}. AI responded: {response}"