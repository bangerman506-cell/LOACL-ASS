import gradio as gr
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import hashlib

# ===========================
# 1. INITIALIZE LLM (CEREBRAS)
# ===========================

def init_llm():
    """Initialize Cerebras LLM"""
    api_key = os.getenv("CEREBRAS_API_KEY")
    
    if not api_key:
        raise Exception("❌ CEREBRAS_API_KEY not found in secrets!")
    
    llm = ChatOpenAI(
        model="llama3.1-70b",
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
        temperature=0.1,
        max_tokens=4000
    )
    
    print("✅ Cerebras LLM initialized")
    return llm

# ===========================
# 2. INITIALIZE RAG (QDRANT)
# ===========================

class SimpleRAG:
    """Simple RAG using Qdrant"""
    
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_url or not qdrant_key:
            print("⚠️ Qdrant not configured - using fallback")
            self.enabled = False
            return
        
        try:
            print("Connecting to Qdrant...")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
            
            print("Loading embedding model...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.collection_name = "code_snippets"
            self.enabled = True
            
            # Create collection if needed
            self._setup_collection()
            
            # Add initial examples
            self._add_initial_examples()
            
            print("✅ Qdrant RAG ready")
            
        except Exception as e:
            print(f"⚠️ Qdrant init failed: {e}")
            self.enabled = False
    
    def _setup_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                print(f"✅ Created collection: {self.collection_name}")
        except Exception as e:
            print(f"Collection setup error: {e}")
    
    def _add_initial_examples(self):
        """Add some initial code examples"""
        try:
            # Check if already has data
            info = self.client.get_collection(self.collection_name)
            if info.points_count > 0:
                print(f"📚 Collection has {info.points_count} examples")
                return
            
            # Initial examples
            examples = [
                """# FastAPI Hello World
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}""",
                
                """# Pandas Data Processing
import pandas as pd

# Read CSV
df = pd.read_csv('data.csv')

# Filter rows
filtered = df[df['age'] > 25]

# Group and aggregate
summary = df.groupby('category')['sales'].sum()

# Save to CSV
filtered.to_csv('output.csv', index=False)""",
                
                """# Requests API Call
import requests

# GET request
response = requests.get('https://api.example.com/data')
data = response.json()

# POST request
payload = {'key': 'value'}
response = requests.post('https://api.example.com/submit', json=payload)

# With error handling
try:
    response = requests.get('https://api.example.com/data', timeout=5)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")""",
                
                """# Async Python Example
import asyncio

async def fetch_data(item_id):
    # Simulate API call
    await asyncio.sleep(1)
    return f"Data for {item_id}"

async def main():
    # Run tasks concurrently
    tasks = [fetch_data(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results

# Run
results = asyncio.run(main())""",
                
                """# File Operations
# Read file
with open('input.txt', 'r') as f:
    content = f.read()

# Write file
with open('output.txt', 'w') as f:
    f.write('Hello World')

# Append to file
with open('log.txt', 'a') as f:
    f.write('New log entry\\n')

# Read lines
with open('data.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        print(line.strip())"""
            ]
            
            # Embed and store
            embeddings = self.embedder.encode(examples).tolist()
            
            points = []
            for i, (example, embedding) in enumerate(zip(examples, embeddings)):
                point_id = int(hashlib.md5(example.encode()).hexdigest()[:8], 16)
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"code": example, "index": i}
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"✅ Added {len(examples)} initial examples")
            
        except Exception as e:
            print(f"Error adding examples: {e}")
    
    def search(self, query, top_k=3):
        """Search for relevant code"""
        if not self.enabled:
            return []
        
        try:
            query_vector = self.embedder.encode([query])[0].tolist()
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            return [hit.payload['code'] for hit in results]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

# ===========================
# 3. CHAT LOGIC
# ===========================

def process_message(user_message, history, llm, rag):
    """Process user message and generate response"""
    
    if not user_message.strip():
        return history
    
    try:
        # Step 1: Search for relevant examples
        print(f"🔍 Searching for: {user_message[:50]}...")
        relevant_examples = rag.search(user_message, top_k=3)
        
        # Step 2: Build context
        context = ""
        if relevant_examples:
            context = "**Relevant examples from knowledge base:**\n\n"
            for i, example in enumerate(relevant_examples, 1):
                context += f"Example {i}:\n```python\n{example}\n```\n\n"
            print(f"✅ Found {len(relevant_examples)} relevant examples")
        else:
            print("ℹ️ No examples found, generating from scratch")
        
        # Step 3: Create prompt
        system_prompt = """You are an expert Python developer and coding assistant.

Your job:
1. Help users write clean, efficient Python code
2. Explain programming concepts clearly
3. Debug and fix code issues
4. Follow best practices

Guidelines:
- Write clean, well-commented code
- Include docstrings and type hints
- Add error handling
- Provide usage examples
- Keep explanations simple and clear"""

        user_prompt = f"""{user_message}

{context}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Step 4: Generate response
        print("🤖 Generating response...")
        response = llm.invoke(messages)
        
        # Step 5: Format response
        assistant_message = response.content
        
        # Add to history
        history.append((user_message, assistant_message))
        
        print("✅ Response generated")
        return history
        
    except Exception as e:
        error_message = f"❌ **Error:** {str(e)}\n\nPlease check:\n1. API keys are set correctly\n2. Qdrant cluster is active\n3. Try again in a moment"
        history.append((user_message, error_message))
        print(f"❌ Error: {e}")
        return history

# ===========================
# 4. INITIALIZE SYSTEM
# ===========================

print("=" * 50)
print("🚀 Starting AI Coding Assistant...")
print("=" * 50)

try:
    llm = init_llm()
    rag = SimpleRAG()
    
    print("\n" + "=" * 50)
    print("✅ System ready!")
    print("=" * 50 + "\n")
    
except Exception as e:
    print(f"\n❌ Initialization failed: {e}\n")
    raise

# ===========================
# 5. GRADIO INTERFACE
# ===========================

def chat(message, history):
    """Wrapper for Gradio"""
    return process_message(message, history, llm, rag)

def check_status():
    """Show system status"""
    status = "**🔧 System Status:**\n\n"
    
    # Check API keys
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    status += f"{'✅' if cerebras_key else '❌'} Cerebras API Key\n"
    status += f"{'✅' if qdrant_url else '❌'} Qdrant URL\n"
    status += f"{'✅' if qdrant_key else '❌'} Qdrant API Key\n"
    status += f"{'✅' if rag.enabled else '❌'} RAG System\n"
    
    if rag.enabled:
        try:
            info = rag.client.get_collection(rag.collection_name)
            status += f"\n📚 Knowledge Base: **{info.points_count} code examples**\n"
        except:
            pass
    
    return status

# Build UI
with gr.Blocks(
    title="AI Coding Assistant",
    theme=gr.themes.Soft(primary_hue="indigo")
    analytics_enabled=False # add this
) as demo:
    
    gr.Markdown("""
    # 🤖 AI Coding Assistant
    
    **Powered by Cerebras (Llama 3.1 70B) + RAG + Qdrant**
    
    Ask me to write code, explain concepts, debug problems, or anything coding-related!
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                value=[],
                height=500,
                show_copy_button=True,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your coding question here...",
                    label="Your Message",
                    scale=4,
                    lines=2
                )
            
            with gr.Row():
                submit = gr.Button("Send 🚀", variant="primary", scale=2)
                clear = gr.Button("Clear Chat 🗑️", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Example Prompts")
            
            examples = gr.Examples(
                examples=[
                    "Create a FastAPI REST API with CRUD operations",
                    "Write a web scraper using BeautifulSoup",
                    "Explain how decorators work in Python",
                    "Create a function to validate email addresses",
                    "Write async code to fetch multiple URLs",
                    "How do I handle errors in Python?"
                ],
                inputs=msg
            )
            
            gr.Markdown("---")
            
            status = gr.Textbox(
                value=check_status(),
                label="System Status",
                lines=10,
                max_lines=15
            )
            
            refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
            refresh_btn.click(fn=check_status, outputs=status)
    
    # Event handlers
    submit.click(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        outputs=msg
    )
    
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        outputs=msg
    )
    
    clear.click(
        fn=lambda: [],
        outputs=chatbot
    )
    
    gr.Markdown("""
    ---
    ### 🎯 Features
    - ✅ Fast inference with Cerebras (fastest LLM API)
    - ✅ RAG-powered code search from knowledge base
    - ✅ Persistent vector database (Qdrant Cloud)
    - ✅ Code examples for Python, FastAPI, Pandas, async, and more
    
    ### 📝 Tips
    - Be specific in your questions
    - Ask for code examples, explanations, or debugging help
    - The system learns from a knowledge base of common patterns
    """)

if name == "main":
import os
port = int(os.getenv("PORT", 10000))
demo.queue()
demo.launch(
server_name="0.0.0.0",
server_port=port,
share=False,
show_error=True
)


