import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

# Load API keys
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------
# Session state
# ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None

def add_message(role, text, model=None):
    st.session_state.messages.append({"role": role, "text": text, "model": model})

def export_chat_log():
    return "\n\n".join(
        [f"{m['role'].upper()} ({m.get('model','')}): {m['text']}" for m in st.session_state.messages]
    )

# ---------------------
# Helper Functions
# ---------------------
def search_company_info(company_name: str) -> str:
    """Get latest news, funding, culture, and recent updates about a company."""
    try:
        search = TavilySearchResults(max_results=5, api_key=TAVILY_KEY, search_depth="advanced")
        query = f"{company_name} company news funding culture recent updates 2024 2025"
        results = search.invoke(query)
        
        content = f"# Company Research: {company_name}\n\n"
        for idx, item in enumerate(results, 1):
            content += f"**Source {idx}:** {item.get('content', '')}\n"
            content += f"URL: {item.get('url', '')}\n\n"
        return content
    except Exception as e:
        return f"Error searching company info: {str(e)}"


def get_interview_questions(role: str, level: str = "mid") -> str:
    """Generate role-specific interview questions with answers."""
    try:
        search = TavilySearchResults(max_results=5, api_key=TAVILY_KEY)
        query = f"{role} {level} level interview questions answers 2024 2025"
        results = search.invoke(query)
        
        content = f"# Interview Questions for {role} ({level} level)\n\n"
        for idx, item in enumerate(results, 1):
            content += f"**Resource {idx}:** {item.get('content', '')}\n\n"
        return content
    except Exception as e:
        return f"Error getting interview questions: {str(e)}"


def salary_research(role: str, location: str) -> str:
    """Get salary ranges and compensation data."""
    try:
        search = TavilySearchResults(max_results=5, api_key=TAVILY_KEY)
        query = f"{role} salary {location} 2024 2025 compensation range total comp"
        results = search.invoke(query)
        
        content = f"# Salary Research: {role} in {location}\n\n"
        for idx, item in enumerate(results, 1):
            content += f"**Source {idx}:** {item.get('content', '')}\n"
            content += f"URL: {item.get('url', '')}\n\n"
        return content
    except Exception as e:
        return f"Error researching salary: {str(e)}"


def resume_tips(role: str, experience: str) -> str:
    """Get resume writing tips and best practices."""
    try:
        search = TavilySearchResults(max_results=5, api_key=TAVILY_KEY)
        query = f"{role} resume tips best practices {experience} 2024 ATS"
        results = search.invoke(query)
        
        content = f"# Resume Tips for {role}\n\n"
        for idx, item in enumerate(results, 1):
            content += f"**Tip {idx}:** {item.get('content', '')}\n\n"
        return content
    except Exception as e:
        return f"Error getting resume tips: {str(e)}"


def industry_trends(industry: str) -> str:
    """Get latest industry trends, hot skills, and market insights."""
    try:
        search = TavilySearchResults(max_results=5, api_key=TAVILY_KEY, search_depth="advanced")
        query = f"{industry} industry trends 2024 2025 hot skills in-demand jobs"
        results = search.invoke(query)
        
        content = f"# Industry Trends: {industry}\n\n"
        for idx, item in enumerate(results, 1):
            content += f"**Insight {idx}:** {item.get('content', '')}\n"
            content += f"URL: {item.get('url', '')}\n\n"
        return content
    except Exception as e:
        return f"Error getting industry trends: {str(e)}"


def initialize_llm(model_choice):
    """Initialize the LLM based on model choice"""
    if model_choice == "OpenAI":
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY, temperature=0.7)
    else:
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=GEMINI_KEY, temperature=0.7)


def process_query(user_query: str, llm):
    """Process user query and return response"""
    
    # Determine which tool to use based on keywords
    query_lower = user_query.lower()
    
    try:
        # Company research
        if any(word in query_lower for word in ["company", "about", "tell me about"]):
            # Extract company name (simple approach)
            words = user_query.split()
            for i, word in enumerate(words):
                if word.lower() in ["about", "company"]:
                    if i + 1 < len(words):
                        company_name = words[i + 1].strip(".,!?")
                        search_results = search_company_info(company_name)
                        
                        prompt = f"""You are a career assistant. Based on this research about {company_name}, 
                        provide a comprehensive summary including recent news, company culture, and any important updates.
                        
                        Research Data:
                        {search_results}
                        
                        Provide a well-structured, informative response."""
                        
                        response = llm.invoke(prompt)
                        return response.content
        
        # Salary research
        elif "salary" in query_lower or "compensation" in query_lower:
            words = user_query.split()
            role = ""
            location = ""
            
            # Simple extraction
            if "for" in query_lower:
                role_start = query_lower.find("for") + 4
                role_end = query_lower.find("in") if "in" in query_lower else len(query_lower)
                role = user_query[role_start:role_end].strip()
            
            if "in" in query_lower:
                loc_start = query_lower.find("in") + 3
                location = user_query[loc_start:].strip("?.,!")
            
            if role and location:
                search_results = salary_research(role, location)
                
                prompt = f"""You are a career assistant. Based on this salary research data, 
                provide a comprehensive salary analysis for {role} in {location}.
                
                Research Data:
                {search_results}
                
                Include base salary ranges, total compensation, and benefits information."""
                
                response = llm.invoke(prompt)
                return response.content
        
        # Interview questions
        elif "interview" in query_lower or "questions" in query_lower:
            words = user_query.split()
            role = ""
            level = "mid"
            
            if "entry" in query_lower:
                level = "entry"
            elif "senior" in query_lower:
                level = "senior"
            
            # Extract role
            for i, word in enumerate(words):
                if word.lower() in ["for", "as"]:
                    if i + 1 < len(words):
                        role = words[i + 1].strip(".,!?")
                        break
            
            if role:
                search_results = get_interview_questions(role, level)
                
                prompt = f"""You are a career assistant. Based on this interview questions research, 
                provide comprehensive interview preparation guidance for {role} at {level} level.
                
                Research Data:
                {search_results}
                
                Include both technical and behavioral questions with sample answers."""
                
                response = llm.invoke(prompt)
                return response.content
        
        # Resume tips
        elif "resume" in query_lower or "cv" in query_lower:
            words = user_query.split()
            role = ""
            experience = ""
            
            # Simple extraction
            for i, word in enumerate(words):
                if word.lower() == "for":
                    if i + 1 < len(words):
                        role = words[i + 1].strip(".,!?")
                if word.lower() in ["years", "experience"]:
                    if i > 0:
                        experience = words[i - 1] + " years"
            
            if role:
                search_results = resume_tips(role, experience)
                
                prompt = f"""You are a career assistant. Based on this resume tips research, 
                provide comprehensive resume guidance for {role}.
                
                Research Data:
                {search_results}
                
                Include formatting advice, key sections, and what recruiters look for."""
                
                response = llm.invoke(prompt)
                return response.content
        
        # Industry trends
        elif "trend" in query_lower or "industry" in query_lower:
            words = user_query.split()
            industry = ""
            
            for i, word in enumerate(words):
                if word.lower() in ["in", "about"]:
                    if i + 1 < len(words):
                        industry = words[i + 1].strip(".,!?")
                        break
            
            if industry:
                search_results = industry_trends(industry)
                
                prompt = f"""You are a career assistant. Based on this industry trends research, 
                provide comprehensive insights about the {industry} industry.
                
                Research Data:
                {search_results}
                
                Include hot skills, emerging technologies, and job market insights."""
                
                response = llm.invoke(prompt)
                return response.content
        
        # Job description analysis
        elif "analyze" in query_lower and "job description" in query_lower:
            prompt = f"""You are a career assistant. Analyze this job description and provide:
            
            1. KEY SKILLS REQUIRED (technical and soft skills)
            2. MUST-HAVE vs NICE-TO-HAVE qualifications
            3. MAIN RESPONSIBILITIES
            4. EXPERIENCE LEVEL required
            5. RESUME TAILORING SUGGESTIONS
            
            Job Description:
            {user_query}
            
            Provide a detailed, structured analysis."""
            
            response = llm.invoke(prompt)
            return response.content
        
        # General career advice
        else:
            prompt = f"""You are a helpful career assistant specializing in job search, interview prep, 
            resume optimization, and career advice. 
            
            User Question: {user_query}
            
            Provide helpful, actionable career advice."""
            
            response = llm.invoke(prompt)
            return response.content
            
    except Exception as e:
        return f"I encountered an error processing your request: {str(e)}\n\nPlease try rephrasing your question or contact support."


def render_chat():
    """Display chat messages"""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg['text'])
        else:
            with st.chat_message("assistant"):
                model_label = f" *({msg.get('model', '')})*" if msg.get("model") else ""
                st.markdown(f"{msg['text']}{model_label}")


# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(
    page_title="Career Assistant 💼", 
    layout="wide", 
    page_icon="💼",
    initial_sidebar_state="expanded"
)

# Header
st.title("💼 Career & Interview Prep Assistant")
st.markdown("*Your AI-powered career companion for job search, interview prep, and salary research*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_choice = st.selectbox(
        "🧠 AI Model",
        ["OpenAI", "Gemini"],
        help="Choose your preferred AI model"
    )
    
    # Reset LLM if model changed
    if st.session_state.current_model != model_choice:
        st.session_state.llm = None
        st.session_state.current_model = model_choice
    
    # Clear chat
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    # Download chat log
    if st.session_state.messages:
        st.download_button(
            "💾 Download Chat Log",
            export_chat_log().encode(),
            file_name="career_assistant_log.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Quick Actions
    st.header("⚡ Quick Actions")
    
    quick_action = st.selectbox(
        "Choose a task:",
        [
            "Custom Question",
            "Research a Company",
            "Get Interview Questions",
            "Analyze Job Description",
            "Salary Research",
            "Resume Tips",
            "Industry Trends"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.info(
        """
        This assistant helps you with:
        - 🏢 Company research
        - 💡 Interview preparation
        - 📄 Resume optimization
        - 💰 Salary insights
        - 📈 Industry trends
        """
    )

# Main content area
st.markdown("### 💬 Chat with Career Assistant")

# Display chat history
if st.session_state.messages:
    render_chat()
else:
    st.info("👋 Hi! I'm your Career Assistant. Ask me anything about job search, interviews, companies, or salaries!")

st.markdown("---")

# Input section based on quick action
if quick_action == "Research a Company":
    with st.form("company_research_form"):
        company_name = st.text_input("🏢 Company Name", placeholder="e.g., Google, Microsoft, Tesla")
        submitted = st.form_submit_button("🔍 Research Company", use_container_width=True)
        
        if submitted and company_name.strip():
            user_query = f"Tell me about {company_name}. Include recent news, company culture, funding, and any important updates."
            add_message("user", user_query)
            
            with st.spinner(f"🔍 Researching {company_name}..."):
                try:
                    if st.session_state.llm is None:
                        st.session_state.llm = initialize_llm(model_choice)
                    
                    answer = process_query(user_query, st.session_state.llm)
                    add_message("assistant", answer, model=model_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif quick_action == "Get Interview Questions":
    with st.form("interview_questions_form"):
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("💼 Job Role", placeholder="e.g., Software Engineer, Product Manager")
        with col2:
            level = st.selectbox("📊 Experience Level", ["entry", "mid", "senior"])
        
        submitted = st.form_submit_button("📝 Get Questions", use_container_width=True)
        
        if submitted and role.strip():
            user_query = f"Give me interview questions for {role} at {level} level. Include both technical and behavioral questions with sample answers."
            add_message("user", user_query)
            
            with st.spinner("📝 Generating interview questions..."):
                try:
                    if st.session_state.llm is None:
                        st.session_state.llm = initialize_llm(model_choice)
                    
                    answer = process_query(user_query, st.session_state.llm)
                    add_message("assistant", answer, model=model_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif quick_action == "Analyze Job Description":
    with st.form("job_description_form"):
        job_desc = st.text_area(
            "📄 Paste Job Description",
            height=200,
            placeholder="Paste the complete job description here..."
        )
        
        submitted = st.form_submit_button("🔍 Analyze JD", use_container_width=True)
        
        if submitted and job_desc.strip():
            user_query = f"Analyze this job description and extract key skills, requirements, responsibilities, and suggest how I should tailor my resume:\n\n{job_desc}"
            add_message("user", "Analyzing job description...")
            
            with st.spinner("🔍 Analyzing job description..."):
                try:
                    if st.session_state.llm is None:
                        st.session_state.llm = initialize_llm(model_choice)
                    
                    answer = process_query(user_query, st.session_state.llm)
                    add_message("assistant", answer, model=model_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif quick_action == "Salary Research":
    with st.form("salary_research_form"):
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("💼 Job Role", placeholder="e.g., Data Scientist")
        with col2:
            location = st.text_input("📍 Location", placeholder="e.g., San Francisco, Remote")
        
        submitted = st.form_submit_button("💰 Research Salary", use_container_width=True)
        
        if submitted and role.strip() and location.strip():
            user_query = f"What is the salary range for {role} in {location}? Include base salary, total compensation, and any benefits information."
            add_message("user", user_query)
            
            with st.spinner("💰 Researching salary data..."):
                try:
                    if st.session_state.llm is None:
                        st.session_state.llm = initialize_llm(model_choice)
                    
                    answer = process_query(user_query, st.session_state.llm)
                    add_message("assistant", answer, model=model_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif quick_action == "Resume Tips":
    with st.form("resume_tips_form"):
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("💼 Target Role", placeholder="e.g., Frontend Developer")
        with col2:
            experience = st.text_input("⏱️ Years of Experience", placeholder="e.g., 3 years")
        
        submitted = st.form_submit_button("📄 Get Resume Tips", use_container_width=True)
        
        if submitted and role.strip():
            user_query = f"Give me resume tips for {role} with {experience} experience. Include formatting advice, key sections, and what recruiters look for."
            add_message("user", user_query)
            
            with st.spinner("📄 Generating resume tips..."):
                try:
                    if st.session_state.llm is None:
                        st.session_state.llm = initialize_llm(model_choice)
                    
                    answer = process_query(user_query, st.session_state.llm)
                    add_message("assistant", answer, model=model_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif quick_action == "Industry Trends":
    with st.form("industry_trends_form"):
        industry = st.text_input("🏭 Industry", placeholder="e.g., AI/ML, Fintech, Healthcare")
        
        submitted = st.form_submit_button("📈 Get Trends", use_container_width=True)
        
        if submitted and industry.strip():
            user_query = f"What are the latest trends in {industry}? Include hot skills, emerging technologies, and job market insights."
            add_message("user", user_query)
            
            with st.spinner("📈 Researching industry trends..."):
                try:
                    if st.session_state.llm is None:
                        st.session_state.llm = initialize_llm(model_choice)
                    
                    answer = process_query(user_query, st.session_state.llm)
                    add_message("assistant", answer, model=model_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

else:  # Custom Question
    with st.form("custom_question_form"):
        custom_prompt = st.text_area(
            "✍️ Your Question",
            height=150,
            placeholder="Ask me anything about careers, interviews, companies, salaries, or resume tips..."
        )
        
        submitted = st.form_submit_button("🚀 Ask Assistant", use_container_width=True)
        
        if submitted and custom_prompt.strip():
            add_message("user", custom_prompt)
            
            with st.spinner("🤖 Thinking..."):
                try:
                    if st.session_state.llm is None:
                        st.session_state.llm = initialize_llm(model_choice)
                    
                    answer = process_query(custom_prompt, st.session_state.llm)
                    add_message("assistant", answer, model=model_choice)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")