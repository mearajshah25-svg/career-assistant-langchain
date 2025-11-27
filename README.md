# 💼 Career & Interview Prep Assistant

Your AI-powered career companion for job search, interview prep, and salary research.

## Features

- 🏢 Company Research
- 💡 Interview Preparation
- 📄 Resume Optimization
- 💰 Salary Insights
- 📈 Industry Trends

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/career-assistant.git
cd career-assistant
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your API keys:
```env
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

## API Keys

Get your API keys from:
- OpenAI: https://platform.openai.com/api-keys
- Gemini: https://makersuite.google.com/app/apikey
- Tavily: https://tavily.com/

## Usage

Run the application:
```bash
streamlit run file.py
```

## Technologies Used

- LangChain
- Streamlit
- OpenAI GPT
- Google Gemini
- Tavily Search

## License

MIT License