import streamlit as st
from docx import Document
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
import re

# Initialize session state
def init_session_state():
    default_values = {
        "need_clarification": False,
        "clarification_query": "",
        "intent": 0,
        "intent_instruction": "",
        "pinecone_context": "",
        "original_query": "",
        "chat_history": []
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Call this function at the very beginning
init_session_state()

# Access your API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "college"

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to the Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# Intent Instructions
INTENT_INSTRUCTIONS = {
    1: """
    You are an AI assistant helping students understand how to declare their major at Texas Tech University. Provide information on the general process, which includes:
    1. Completing the Academic Transfer Form with the advisor for the new major
    2. Using Raider Success Hub to find an appointment with the new advisor
    3. Selecting 'Change My Major' option in Raider Success Hub
    Emphasize that this is the general process and some majors may have additional requirements.
    """,
    2: """
    You are an AI assistant informing students about special requirements for declaring majors in specific colleges or departments at Texas Tech University. When asked about a particular college or department, provide the following information:
    1. Rawls College of Business: Direct students to check the specific requirements
    2. Whitacre College of Engineering: Inform students to review requirements 
    3. Biological Sciences, College of Arts & Sciences: Guide students to check requirements 
    4. Physics and Astronomy, College of Arts and Sciences: Advise students to look at requirements 
    5. Wind Energy, College of Arts and Sciences: Direct students to review requirements.
    Emphasize that these links provide the most up-to-date and detailed information for each specific college or department.
    """,
    3: """
    You are an AI assistant guiding students on how to use the Raider Success Hub at Texas Tech University for declaring a major. Provide the following information:
    1. Explain that Raider Success Hub is the platform used to schedule appointments with advisors for changing majors
    2. Instruct students to access Raider Success Hub.
    3. Guide them to select the 'Change My Major' option within the platform
    4. Emphasize that this is the official method for scheduling appointments related to changing majors
    """,
    4: """
    You are an AI assistant informing students about GPA and course requirements for declaring majors at Texas Tech University. Provide the following information:
    1. Explain that some majors have higher GPA requirements than others
    2. Mention that certain majors may have specific completed course requirements before students can declare
    3. Emphasize the importance of checking the specific requirements for their intended major
    4. Advise students to consult with an academic advisor or check the department's website for the most accurate and up-to-date information on GPA and course requirements
    5. Remind students that meeting the minimum requirements does not guarantee acceptance into a major, as some programs may have limited capacity
    """,
    5: """
    You are an AI assistant helping students with general queries about declaring majors at Texas Tech University. 
    Provide helpful information based on the context available, and if the query is outside your knowledge base, 
    advise the student to contact an academic advisor for more specific information.
    """,
    6: """
    You are an AI assistant providing comprehensive information and guidance about academic advising for students at Texas Tech University. Cover the following key aspects:
    1. Preparation for advising appointments
    2. Important tools and resources for students, such as Raider Success Hub and Degree Works
    3. Navigation of Raiderlink, the university's online portal
    4. Information on course registration and enrollment management
    5. Details about financial management, including tuition payments and financial aid
    6. Academic resources and tools for exploring course options
    7. Information on transcripts, grades, and academic calendars
    Emphasize that this information serves as a reference guide to help students understand the advising process, navigate university systems, and make informed decisions about their academic journey at Texas Tech University.
    """,
    7: """
    You are an AI assistant providing information to new students at Texas Tech University about orientation and initial advising. Cover the following key aspects:
    1. The mission of University Advising
    2. Information about Red Raider Orientation (RRO)
    3. The eXplore program
    4. How to contact University Advising for questions
    Emphasize that attending RRO is where new students will meet with an advisor and register for their first semester of classes.
    """,
    8: """
    You are an AI assistant providing information about courses, majors, and degree programs at Texas Tech University. Cover the following key aspects:
    1. Available majors and degree programs
    2. Specific information about a requested major or program
    3. Degree types (B.A., B.S., B.B.A., etc.) and their meanings
    4. Online program options
    5. Concentrations within majors
    Emphasize the diversity of programs available and direct students to the official catalog for the most up-to-date and detailed information.
    """,
    9: """
    You are an AI assistant providing comprehensive information about student life and wellness at Texas Tech University. Cover the following key aspects:
    1. Staying healthy and maintaining fitness
    2. Managing finances and saving money
    3. Campus events and activities
    4. Organization and time management
    5. Preparing for classes and academic success
    6. Dining options and nutrition
    7. Dorm life and essentials
    8. Campus safety
    9. Transportation
    10. Career development
    11. Technology and IT services
    Provide practical tips and guidance to help students navigate their college experience successfully.
    """,
    10: """
    You are an AI assistant providing specific information about food, dining, and nutrition for students at Texas Tech University. Cover the following key aspects:
    1. On-campus dining options and locations
    2. Smart Choice dining locations and healthy eating options
    3. Meal planning and budgeting tips for students
    4. Nutritional advice for maintaining a balanced diet in college
    5. Dorm cooking ideas and recipes
    6. Information about meal plans and dining dollars
    7. Special dietary accommodations (vegetarian, vegan, gluten-free, etc.)
    8. Tips for eating healthy on a student budget
    9. Local off-campus dining options near Texas Tech
    10. Food safety and storage tips for dorm living
    Provide practical, actionable advice to help students make informed decisions about their dining and nutrition while at Texas Tech University.
    """
}

# Helper functions
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def truncate_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def upsert_to_pinecone(text, file_name, file_id):
    chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]  # Split into 8000 character chunks
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        metadata = {
            "file_name": file_name,
            "file_id": file_id,
            "chunk_id": i,
            "chunk_text": chunk
        }
        index.upsert(vectors=[(f"{file_id}_{i}", embedding, metadata)])
        time.sleep(1)  # To avoid rate limiting

def query_pinecone(query, top_k=5):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results['matches']:
        if 'chunk_text' in match['metadata']:
            contexts.append(match['metadata']['chunk_text'])
        else:
            contexts.append(f"Content from {match['metadata'].get('file_name', 'unknown file')}")
    return " ".join(contexts)

def identify_intent(query):
    intent_prompt = f"""
    Identify the primary intent of this query related to academic advising or student life at Texas Tech University. Choose from the following intents:
    1. General Process for Declaring a Major
    2. Special Requirements for Specific Colleges/Departments
    3. Using Raider Success Hub
    4. GPA and Course Requirements
    5. Other General Queries about Declaring Majors
    6. Comprehensive Academic Advising Information
    7. New Student Orientation and Initial Advising
    8. Course, Major, and Degree Program Information
    9. Student Life and Wellness
    10.Food, Dining, and Nutrition

    Respond with ONLY the number (1-10) of the most relevant intent.

    Query: {query}
    """
    intent_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an intent identification assistant specializing in queries about  Texas Tech University."},
            {"role": "user", "content": intent_prompt}
        ]
    )
    response_text = intent_response.choices[0].message.content.strip()
    
    # Extract the first number from the response
    intent_number = re.search(r'\d+', response_text)
    if intent_number:
        return int(intent_number.group())
    

def get_intent_instruction(intent):
    return INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS[5])

def generate_clarification_query(intent):
    intent_options = {
        1: ["Steps to declare a major", "Required forms", "Advisor meeting process", "Timeline for declaration"],
        2: ["Business school requirements", "Engineering prerequisites", "Arts & Sciences specific rules", "Minimum GPA for specific colleges"],
        3: ["How to access Raider Success Hub", "Scheduling advisor appointments", "Navigating the platform", "Technical support for Raider Success Hub"],
        4: ["Minimum GPA requirements", "Required core courses", "Credit hour prerequisites", "Transfer credit considerations"],
        5: ["Changing majors process", "Double major requirements", "Minor declaration process", "Interdisciplinary studies options"],
        6: ["Preparation for advising appointments", "Important academic tools and resources", "Course registration process", "Financial management information"],
        7: ["Red Raider Orientation details", "University Advising mission", "eXplore program information", "Contacting University Advising"],
        8: ["Specific major information", "Degree types explanation", "Online program options", "Concentrations within majors", "General list of available majors"],
        9: [ "Financial management", "Campus activities", "Study strategies", "Dorm life essentials"],
        10:  [ "Healthy eating tips", "On-campus dining options","Meal planning and budgeting",  "Smart Choice dining locations","Dorm cooking ideas" ]
    }
    
    options = intent_options.get(intent, ["General information", "Specific examples", "Common issues", "Best practices"])
    
    clarification_prompt = f"""
    To better assist you with your query about {INTENT_INSTRUCTIONS[intent].split('.')[0].strip()}, could you please specify what aspect you're most interested in? 

    You can choose from options like:
    1. {options[0]}
    2. {options[1]}
    3. {options[2]}
    4. {options[3]}
    5. {options[4] if len(options) > 4 else "Other specific information"}

    Or feel free to ask about any other specific information you need.
    """
    return clarification_prompt

def process_clarification(original_query, clarification):
    combined_query = f"{original_query} {clarification}"
    clarification_embedding = get_embedding(combined_query)
    clarification_context = query_pinecone(combined_query, top_k=3)
    return clarification_context

def generate_final_response(query, intent, intent_instruction, combined_context, user_clarification=None):
    response_prompt = f"""
    Original Query: {query}
    Intent: {intent}
    System Instruction: {intent_instruction}
    Context: {combined_context}
    User Clarification: {user_clarification if user_clarification else 'No clarification provided'}

    Generate a comprehensive response based on the above information, paying special attention to the clarification context.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are College Buddy, an AI assistant designed to help students with their academic queries at Texas Tech University."},
            {"role": "user", "content": response_prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def process_query(query):
    try:
        intent = identify_intent(query)
        st.session_state.need_clarification = True
        st.session_state.clarification_query = generate_clarification_query(intent)
        st.session_state.original_query = query
        st.session_state.intent = intent
        return None
    except Exception as e:
        st.error(f"An error occurred while processing your query: {str(e)}")
        st.session_state.need_clarification = False
        return "I'm sorry, but I encountered an error while processing your query. Could you please try rephrasing your question?"

# Function to save chat history
def save_chat_history(query, answer):
    st.session_state.chat_history.append({"query": query, "answer": answer})

# Function to display chat history
def display_chat_history():
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, chat in enumerate(st.session_state.chat_history):
            st.text(f"Q{i+1}: {chat['query']}")
            st.text(f"A{i+1}: {chat['answer']}")
            st.markdown("---")

def handle_error(error):
    st.error(f"An error occurred: {str(error)}")
    


# Streamlit Interface
st.set_page_config(page_title="College Buddy Assistant", layout="wide")
st.title("College Buddy Assistant")
st.markdown("Welcome to College Buddy! I am here to help you stay organized, find information fast and provide assistance with academic advising at Texas Tech University. Feel free to ask me a question below.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload the Word Documents (DOCX)", type="docx", accept_multiple_files=True)
    if uploaded_files:
        total_token_count = 0
        for uploaded_file in uploaded_files:
            file_id = str(uuid.uuid4())
            text = extract_text_from_docx(uploaded_file)
            token_count = num_tokens_from_string(text)
            total_token_count += token_count
            # Upsert to Pinecone
            upsert_to_pinecone(text, uploaded_file.name, file_id)
            st.text(f"Uploaded: {uploaded_file.name}")
            st.text(f"File ID: {file_id}")
        st.text(f"Total token count: {total_token_count}")

# Modify the main content area
st.header("Ask Your Question")
user_query = st.text_input("What would you like to know about academic advising or declaring your major?")

if st.button("Get Answer"):
    if user_query:
        try:
            with st.spinner("Analyzing query..."):
                result = process_query(user_query)
                if result is None:
                    st.subheader("Follow-up Question:")
                    st.write(st.session_state.clarification_query)
                else:
                    st.write(result)
        except Exception as e:
            handle_error(e)
    else:
        st.warning("Please enter a question before searching.")

if st.session_state.need_clarification:
    clarification_input = st.text_input("Your response:", key="clarification_input")
    if st.button("Submit Clarification"):
        if clarification_input:
            try:
                with st.spinner("Generating final response..."):
                    intent_instruction = get_intent_instruction(st.session_state.intent)
                    original_context = query_pinecone(st.session_state.original_query)
                    clarification_context = process_clarification(st.session_state.original_query, clarification_input)
                    combined_context = f"Original context: {original_context}\nClarification context: {clarification_context}"
                    final_answer = generate_final_response(
                        st.session_state.original_query, 
                        st.session_state.intent,
                        intent_instruction, 
                        combined_context, 
                        clarification_input
                    )
                    st.subheader("Answer:")
                    st.write(final_answer)
                    save_chat_history(f"{st.session_state.original_query} (Clarification: {clarification_input})", final_answer)
                    st.session_state.need_clarification = False
                # Removed collect_feedback() call
            except Exception as e:
                handle_error(e)
        else:
            st.warning("Please provide a clarification before submitting.")
