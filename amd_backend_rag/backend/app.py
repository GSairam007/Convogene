from flask import Response
from flask import Flask, request, jsonify, abort, make_response
from flask_cors import CORS
from flask_socketio import SocketIO, Namespace
from utils.tools import google_web_search
from utils.utils import format_docs, format_web_docs, get_vectorstore, rerank_docs, grading
from utils.history import collection, append_chat_entry, get_chat_history_by_date, update_feedback, get_feedback_by_sentiment
from dotenv import load_dotenv
import os, logging
from langchain_openai import ChatOpenAI
import time
import json
from bson import json_util
import warnings
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import pytz
from langsmith import Client
from flask import Flask
import re
from threading import Thread, Lock
import requests

load_dotenv()
warnings.filterwarnings('ignore', message="Valid config keys have changed in V2:")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "https://aryan-convogene-test.ashycliff-b1403732.eastus.azurecontainerapps.io",
    "https://amd-frontend-prod.ashycliff-b1403732.eastus.azurecontainerapps.io",
    "https://amd-frontend-test-ssl.ashycliff-b1403732.eastus.azurecontainerapps.io"
], "methods": ["*"]}}, allow_headers=["Content-Type"])

socketio = SocketIO(app, cors_allowed_origins=[
    "https://aryan-convogene-test.ashycliff-b1403732.eastus.azurecontainerapps.io",
    "https://amd-frontend-prod.ashycliff-b1403732.eastus.azurecontainerapps.io",
    "https://amd-frontend-test-ssl.ashycliff-b1403732.eastus.azurecontainerapps.io"
])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "convogene-amd"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = "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"
LANGCHAIN_PROJECT = "convogene-amd"
TARGET_TIMEZONE = pytz.timezone('Asia/Kolkata')

IST = pytz.timezone('Asia/Kolkata')
logger.info("Start time: %s, End time: %s", datetime.now(IST) - timedelta(days=1), datetime.now(IST))

# IN-MEMORY CACHE & SNAPSHOT CONFIG
CACHE_DURATION = 60  # seconds
SNAPSHOT_FILE = "snapshot.json"

DATA_CACHE = {
    "data": None,
    "analytics": None,
    "last_updated": None
}
CACHE_LOCK = Lock()

# Constants for Costs
COHERE_TOKEN_PRICE_input = 0.0000025
COHERE_TOKEN_PRICE_output = 0.00001
CHATOPENAI_TOKEN_PRICE_input = 0.0000025
CHATOPENAI_TOKEN_PRICE_output = 0.00001
RERANKING_COST = 0.002


def fetch_langsmith_data():
    """Fetches runs from LangSmith for the last 24 hours."""
    end_time_ist = datetime.now(TARGET_TIMEZONE)
    start_time_ist = end_time_ist - timedelta(days=1)
    end_time_utc = end_time_ist.astimezone(pytz.UTC)
    start_time_utc = start_time_ist.astimezone(pytz.UTC)

    client = Client(api_key=LANGCHAIN_API_KEY)
    runs = list(client.list_runs(
        project_name=LANGCHAIN_PROJECT,
        start_time=start_time_utc,
        end_time=end_time_utc
    ))
    return runs

def process_data_payload(runs):
    """Processes runs to generate the /data API payload."""
    names = []
    run_types = []
    rows = []
    
    question_aggregates = defaultdict(lambda: {
        "Question": None,
        "Name": "ChatOpenAI",
        "Time": None,
        "Total Tokens": 0,
        "Total Cost($)": 0.0,
        "Total Cost(rs)": 0.0,
        "Runs": []
    })

    for run in runs:
        # Extract input and identify the question using regex
        try:
            # Check safely for input content structure
            if isinstance(run.inputs, dict) and 'messages' in run.inputs and len(run.inputs['messages']) > 0:
                 # Standard structure
                 msg_content = run.inputs['messages'][0][0]['kwargs']['content']
            elif isinstance(run.inputs, dict) and 'input' in run.inputs:
                 msg_content = run.inputs['input']
            else:
                 msg_content = str(run.inputs)
                 
            question_match = re.search(r"Question: (.*?)(?:\n|$)", msg_content)
            question = question_match.group(1).strip() if question_match else "Question not found"
        except Exception as e:
            question = "Unknown Question"

        # Convert start time to IST
        if run.start_time:
            utc_time = run.start_time.replace(tzinfo=timezone.utc)
            time_ist = utc_time.astimezone(TARGET_TIMEZONE).strftime("%d, %b, %Y, %H:%M:%S")
        else:
            time_ist = None

        entry = question_aggregates[question]
        entry["Question"] = question
        entry["Time"] = entry["Time"] or time_ist

        run_cost = (
            (run.prompt_tokens * (COHERE_TOKEN_PRICE_input if run.name == 'ChatCohere' else CHATOPENAI_TOKEN_PRICE_input)) +
            (run.completion_tokens * (COHERE_TOKEN_PRICE_output if run.name == 'ChatCohere' else CHATOPENAI_TOKEN_PRICE_output))
        )

        if len(entry["Runs"]) < 12:
            entry["Runs"].append({
                "Time": time_ist,
                "Tokens": run.total_tokens,
                "Cost($)": round(run_cost, 6)
            })
        else:
            continue

        entry["Total Tokens"] = sum(r['Tokens'] for r in entry["Runs"])
        entry["Total Cost($)"] = round(sum(r['Cost($)'] for r in entry["Runs"]), 6)
        entry["Total Cost(rs)"] = round(entry["Total Cost($)"] * 84, 3)

        names.append(run.name)
        run_types.append(run.run_type)

        run_data = {}
        if run.error is None or run.end_time is None:
            run_data['Status'] = '✅'
        else:
            run_data['Status'] = '❗'
        rows.append(run_data)

    output_data = pd.DataFrame.from_records([
        {k: v for k, v in question_data.items() if k != 'Runs'}
        for question_data in question_aggregates.values()
    ])

    bar_chart_data = [] # Default
    table_data = []

    if not output_data.empty:
        output_data['Time'] = pd.to_datetime(output_data['Time'], errors='coerce')
        output_data.dropna(subset=['Time'], inplace=True)
        output_data.set_index('Time', inplace=True)
        
        try:
            bar_chart_data_df = output_data.resample('h').size().reset_index(name='Number of Requests')
            bar_chart_data_df.rename(columns={"Time": "name", "Number of Requests": "value"}, inplace=True)
            bar_chart_data_df['name'] = bar_chart_data_df['name'].dt.strftime("%Y-%m-%d %H:%M:%S") 
            bar_chart_data = bar_chart_data_df.to_dict(orient='records')
        except Exception:
             pass

        table_data = output_data.reset_index().to_dict(orient='records')
        for item in table_data:
            if isinstance(item['Time'], (pd.Timestamp, datetime)):
                 item['Time'] = item['Time'].strftime("%d, %b, %Y, %H:%M:%S")

        table_data = sorted(table_data, key=lambda x: datetime.strptime(str(x["Time"]), "%d, %b, %Y, %H:%M:%S") if isinstance(x["Time"], str) else x["Time"], reverse=True)

    names_count = pd.Series(names).value_counts().to_dict()
    keys_to_keep = {'ChatOpenAI', 'ChatCohere'}
    pie_chart_data_names_count = [{"name": name, "value": count} for name, count in names_count.items() if name in keys_to_keep]

    run_types_count = pd.Series(run_types).value_counts().to_dict()
    pie_chart_data_run_types_count = [{"name": run_type, "value": count} for run_type, count in run_types_count.items()]

    df3 = pd.DataFrame(rows)
    if not df3.empty:
        status_counts = df3['Status'].value_counts().to_dict()
        pie_chart_data_status = [{"name": status, "value": count} for status, count in status_counts.items()]
    else:
        pie_chart_data_status = []

    payload = {
        "pieChartDataStatus": pie_chart_data_status,
        "pieChartDataNamesCount": pie_chart_data_names_count,
        "pieChartDataRunTypesCount": pie_chart_data_run_types_count,
        "tableData": table_data,
        "barChartData": bar_chart_data 
    }
    return payload

def process_analytics_payload(runs):
    """Processes runs to generate the /analytics API payload."""
    total_cost = 0
    t_count = 0
    
    # Local DataFrame for calculations
    df_local = pd.DataFrame(columns=['TotalTokens', 'TotalCost'])

    def convert_to_timezone(utc_timestamp):
        if utc_timestamp:
            try:
                utc_time = datetime.strptime(utc_timestamp, "%Y-%m-%dT%H:%M:%S.%f")
            except ValueError:
                utc_time = datetime.strptime(utc_timestamp, "%Y-%m-%dT%H:%M:%S")

            utc_zone = pytz.timezone('UTC')
            localized_time = utc_zone.localize(utc_time).astimezone(TARGET_TIMEZONE)
            return localized_time.isoformat()
        return None

    grouped_runs = defaultdict(lambda: {
        'cohere_input_tokens': 0,
        'cohere_output_tokens': 0,
        'cohere_cost': 0,
        'chatopenai_input_tokens': 0,
        'chatopenai_output_tokens': 0,
        'chatopenai_cost': 0
    })

    for run in runs:
        timestamp = convert_to_timezone(run.start_time.isoformat()) if run.start_time else None
        
        if run.name == 'ChatCohere':
            input_tokens = run.prompt_tokens if run.prompt_tokens else 0
            output_tokens = run.completion_tokens if run.completion_tokens else 0
            token_cost = (input_tokens * COHERE_TOKEN_PRICE_input) + (output_tokens * COHERE_TOKEN_PRICE_output)

            grouped_runs[timestamp]['cohere_input_tokens'] += input_tokens
            grouped_runs[timestamp]['cohere_output_tokens'] += output_tokens
            grouped_runs[timestamp]['cohere_cost'] += token_cost

        elif run.name == 'ChatOpenAI':
            input_tokens = run.prompt_tokens if run.prompt_tokens else 0
            output_tokens = run.completion_tokens if run.completion_tokens else 0
            token_cost = (input_tokens * CHATOPENAI_TOKEN_PRICE_input) + (output_tokens * CHATOPENAI_TOKEN_PRICE_output)

            grouped_runs[timestamp]['chatopenai_input_tokens'] += input_tokens
            grouped_runs[timestamp]['chatopenai_output_tokens'] += output_tokens
            grouped_runs[timestamp]['chatopenai_cost'] += token_cost
            t_count += 1 

    batch_size = 12
    grouped_data = list(grouped_runs.items())
    
    for i in range(0, len(grouped_data), batch_size):
        batch = grouped_data[i:i+batch_size]
        total_input_tokens = 0
        TC = 0

        for _, data in batch:
            total_input_tokens += data['cohere_input_tokens'] + data['chatopenai_input_tokens']
            total_cost += data['cohere_cost'] + data['chatopenai_cost'] + RERANKING_COST
            TC += data['cohere_cost'] + data['chatopenai_cost'] + RERANKING_COST

        new_row = pd.DataFrame([{
            'TotalTokens': total_input_tokens,
            'TotalCost': TC
        }])
        df_local = pd.concat([df_local, new_row], ignore_index=True)

    average_tokens_per_question = df_local['TotalTokens'].mean() if not df_local.empty else 0
    average_cost_per_question = df_local['TotalCost'].mean() if not df_local.empty else 0
    average_latency_per_question = 5.05 

    payload = {
        "averageTokensPerQuestion": average_tokens_per_question,
        "averageCostPerQuestion": round(average_cost_per_question, 4),
        "averageLatencyPerQuestion": average_latency_per_question,
        "totalCost": round(total_cost, 4),
        "totalQuestions": t_count // 2, 
        "execution_time_seconds": 0 
    }
    return payload

def load_snapshot_to_cache():
    """Loads logic from snapshot.json to memory."""
    try:
        if os.path.exists(SNAPSHOT_FILE):
            with open(SNAPSHOT_FILE, 'r') as f:
                snapshot = json.load(f)
                with CACHE_LOCK:
                    DATA_CACHE["data"] = snapshot.get("data")
                    DATA_CACHE["analytics"] = snapshot.get("analytics")
                    DATA_CACHE["last_updated"] = snapshot.get("timestamp")
            logger.info("Loaded snapshot from disk.")
            return True
    except Exception as e:
        logger.error(f"Error loading snapshot: {e}")
    return False

def update_snapshot():
    """Fetches new data, saves to disk, updates cache."""
    try:
        logger.info("📸 SNAPSHOT: Starting update...")
        start_time = time.time()
        
        runs = fetch_langsmith_data()
        
        data_payload = process_data_payload(runs)
        analytics_payload = process_analytics_payload(runs)
        
        snapshot = {
            "timestamp": time.time(),
            "data": data_payload,
            "analytics": analytics_payload
        }
        
        # Save to disk
        with open(SNAPSHOT_FILE, 'w') as f:
            json.dump(snapshot, f)
            
        # Update memory
        with CACHE_LOCK:
            DATA_CACHE["data"] = data_payload
            DATA_CACHE["analytics"] = analytics_payload
            DATA_CACHE["last_updated"] = time.time()
            
        logger.info(f"📸 SNAPSHOT: Update completed in {time.time() - start_time:.2f}s")
        return True
    except Exception as e:
        logger.error(f"📸 SNAPSHOT: Update failed: {e}")
        return False

@app.route("/data", methods=['POST', 'GET', 'OPTIONS', 'PUT', 'DELETE', 'PATCH'])
def data():
    # Strict read from cache/snapshot
    with CACHE_LOCK:
        if DATA_CACHE["data"] is None:
            # Try loading from disk if memory is empty
            if not load_snapshot_to_cache():
                 # Trigger an immediate background update if completely empty?
                 # Or just return empty/loading state.
                 pass
            
            # Use data if loaded
            if DATA_CACHE["data"]:
                 return jsonify(DATA_CACHE["data"]), 200
            
            return jsonify({"error": "Data initializing..."}), 503
        
        return jsonify(DATA_CACHE["data"]), 200

@app.route("/analytics", methods=['POST', 'GET', 'OPTIONS', 'PUT', 'DELETE', 'PATCH'])
def analytics():
    # Strict read from cache/snapshot
    st_time = time.time()
    with CACHE_LOCK:
       if DATA_CACHE["analytics"] is None:
            if not load_snapshot_to_cache():
                 # Attempt load
                 pass
            if not DATA_CACHE["analytics"]:
                return jsonify({"error": "Data initializing..."}), 503
       
       # Return copy with execution time
       cached = dict(DATA_CACHE["analytics"])
       
    cached["execution_time_seconds"] = round(time.time() - st_time, 4)
    return jsonify(cached), 200


"""# ---------------------------------------------------Chat API---------------------------------------------------------"""
class QueryNamespace(Namespace):
    def on_connect(self):
        print("Client connected to /query namespace")

    def on_disconnect(self):
        print("Client disconnected from /query namespace")

socketio.on_namespace(QueryNamespace("/openai"))

def rag_qa_stream(prompt):
    global global_time_to_first_response, global_total_process_time
    try:
        source_origin = ""
        full_response = ""
        first_chunk_logged = False
        sources = []

        process_start_time = time.time()
        logger.info(f"Received prompt: {prompt}")

        # Retrieve documents from vector store
        try:
            vectorstore = get_vectorstore()
            retrieval_start_time = time.time()
            docs = vectorstore.similarity_search_with_score(prompt, k=20)
            fetching_time = time.time() - retrieval_start_time
            logger.info(f"Data fetched in {fetching_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            yield b"Error retrieving documents"
            return

        # Rerank documents
        try:
            rerank_start_time = time.time()
            retrieved_docs = [doc for doc, _ in docs]
            reranked_docs = rerank_docs(prompt, retrieved_docs)
            rerank_time = time.time() - rerank_start_time
            logger.info(f"Rerank done in {rerank_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during document reranking: {str(e)}")
            yield b"Error reranking documents"
            return

        # Grading process
        try:
            grading_start_time = time.time()
            approved_docs = grading(reranked_docs, prompt)
            print(f"approved {len(approved_docs)} docs")
            grading_time = time.time() - grading_start_time
            logger.info(f"Grading done in {grading_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error during document grading: {str(e)}")
            yield b"Error grading documents"
            return

        # Check if additional Google search is needed
        try:
            if len(approved_docs) < 5:
                if len(approved_docs) != 0:
                    source_origin = "Web & amd.VectorDB"
                else:
                    source_origin = "web"
                search_start_time = time.time()
                links, content = google_web_search(prompt, 10 - len(approved_docs))
                logger.info(f"Google search done in {time.time() - search_start_time:.2f} seconds")
                formatted_docs, sources = format_docs(approved_docs)
                approved_docs = formatted_docs + format_web_docs(content)
                sources.extend(links)
            else:
                approved_docs, sources = format_docs(approved_docs)
                source_origin = "amd.VectorDB"

            sources = list(set(sources))
            print(source_origin)
        except Exception as e:
            logger.error(f"Error during Google search: {str(e)}")
            yield b"Error fetching additional context"
            return

        sources = sorted(sources[:3])
        print(sources)
        relevant_docs = approved_docs

        try:
            prompt_template = f"""
            You are a helpful and informative AI assistant. Your task is to answer the user's question based on the provided context.

            Context: {relevant_docs}

            Question: {prompt}

            Instructions:

            1. Answer the question truthfully and comprehensively, relying solely on the given context.
            2. Do not fabricate information or answer from your own knowledge base and don't talk about what is in the context.
            3. Cite the source documents (as clickable) that support your answer by providing their URLs mentioned here and don't give any placeholders for urls, sources = {sources}. 

            [Provide your answer here]

            **Sources:**

            [List the URLs as urls of the source documents in bullet format used to answer the question, one clickable URL per line](sources = {sources})

            Print the below Origin as it mentioned below
            **Origin:** {source_origin} 
            """

            llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model_name="gpt-4o", temperature=0.6, streaming=True)
            print(llm.model_name)
            response_start_time = time.time()
            for chunk in llm.stream(prompt_template):
                chunk_text = chunk.content
                full_response += chunk_text

                if not first_chunk_logged:
                    first_chunk_time = time.time()
                    global_time_to_first_response = fetching_time + rerank_time + grading_time + (first_chunk_time - response_start_time)
                    logger.info(f"Time to first response: {global_time_to_first_response:.2f} seconds")
                    first_chunk_logged = True

                yield chunk_text.encode('utf-8')
                socketio.emit('response', {'text': chunk_text}, namespace="/openai")

            append_chat_entry(prompt, full_response)

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            yield b"Error generating response"
            return

    except Exception as e:
        logger.error(f"Error in overall process: {str(e)}")

@app.route('/rag_qa_api_stream', methods=['POST', 'GET', 'OPTIONS', 'PUT', 'DELETE', 'PATCH'])
def rag_qa_api_stream():
    data = request.json
    prompt = data.get('text')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    return Response(rag_qa_stream(prompt), mimetype='text/plain')

"""---------------------------------------------------------History-----------------------------------------------"""
@app.route('/list_files', methods=['POST', 'GET', 'OPTIONS', 'PUT', 'DELETE', 'PATCH'])
def list_files():
    try:
        files = collection.distinct('date')
        files.sort(reverse=True)
        return jsonify(files)
    except Exception as e:
        logger.error(f"Error retrieving file list: {str(e)}")
        return jsonify({"error": "Failed to retrieve file list"}), 500

@app.route('/one_file', methods=['POST', 'GET'])
def one_file():
    try:
        data = request.get_json()
        file_date = data.get('file')
        if not file_date:
            return jsonify({"error": "File parameter is required"}), 400

        chat_history = get_chat_history_by_date(file_date)
        if chat_history:
            return json.loads(json_util.dumps(chat_history))
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return jsonify({"error": "Failed to retrieve data"}), 500

"""----------------------------------Related questions-----------------------------------------------"""
@app.route('/related_questions', methods=['POST'])
def related_questions():
    data = request.get_json()
    prompt = data.get('prompt')
    answer = data.get('answer')

    if not prompt or not answer:
        return jsonify({'error': 'Both prompt and answer are required'}), 400

    try:
        related_questions = generate_related_questions(prompt, answer)
        return jsonify({'related_questions': related_questions})
    except Exception as e:
        logger.error(f"Error generating related questions: {str(e)}")
        return jsonify({"error": "Failed to generate related questions"}), 500

def generate_related_questions(prompt, answer):
    try:
        llm = ChatOpenAI(api_key=os.environ.get('OPENAI_API_KEY'), temperature=0.7, model_name="gpt-4o")
        prompt_template = f"""
        Question: {prompt}
        Answer: {answer}

        generate four related AMD questions based on the **Question** and **Answer**, one per line. Do not number the questions. 
        """

        response = llm.invoke(prompt_template)
        related_questions = [line.strip() for line in response.content.splitlines() if line.strip()]
        logger.info(f"Generated related questions: {related_questions}")
        return related_questions[:4]
    except Exception as e:
        logger.error(f"Error generating related questions: {str(e)}")
        return []

"""--------------------------------------------Feedback--------------------------------------------"""
@app.route('/fetch_feedback', methods=['POST'])
def fetch_feedback():
    data = request.get_json()
    update_feedback(data['Query'], data['Response'], data)
    return {
        "message": "Feedback received successfully",
        "data": data,
        "code": "200"
    }

@app.route('/fetch_feedback_text', methods=['POST'])
def fetch_feedback_text():
    data = request.get_json()
    update_feedback(data["Query"], data['Response'], data)
    logger.info("Feedback for query: %s", data["Query"])
    return {
        "message": "Feedback received successfully",
        "data": data,
        "code": "200"
    }

@app.route('/filter_feedback', methods=['POST'])
def filter_feedback():
    try:
        data = request.get_json()
        sentiment = data.get("sentiment")
        file_date = data.get("file")

        if not file_date:
            return jsonify({"error": "File parameter is required"}), 400
        if sentiment not in ["Positive", "Negative"]:
            return jsonify({"error": "Invalid sentiment value"}), 400

        chat_history = get_feedback_by_sentiment(file_date, sentiment)

        if chat_history:
            clean_records = json.loads(json_util.dumps(chat_history))
            filtered_entries = []
            for record in clean_records:
                for entry in record.get("entries", []):
                    filtered_entries.append(entry)

            return jsonify(filtered_entries)
        else:
            return jsonify([])
    except Exception as e:
        logger.error(f"Error retrieving feedback: {str(e)}")
        return jsonify({"error": "Failed to retrieve data"}), 500

"""-------------------------------- CACHE STATUS & CONTROL --------------------------------"""

@app.route("/cache/status", methods=['GET'])
def cache_status():
    """Check cache status and age"""
    with CACHE_LOCK:
        data_cached = DATA_CACHE.get('data') is not None
        analytics_cached = DATA_CACHE.get('analytics') is not None

        last_updated = DATA_CACHE.get('last_updated')
        age = time.time() - last_updated if last_updated else None

        return jsonify({
            "cached": data_cached and analytics_cached,
            "age_seconds": round(age, 2) if age else None,
            "age_minutes": round(age / 60, 2) if age else None,
            "fresh": age < CACHE_DURATION if age else False,
            "cache_duration_seconds": CACHE_DURATION
        })

@app.route("/cache/clear", methods=['POST'])
def cache_clear():
    """Manually clear cache (useful for debugging)"""
    with CACHE_LOCK:
        DATA_CACHE['data'] = None
        DATA_CACHE['analytics'] = None
        DATA_CACHE['last_updated'] = None

    logger.info(" Cache cleared manually")
    return jsonify({"message": "Cache cleared successfully"})

@app.route("/cache/refresh", methods=['POST'])
def cache_refresh():
    """Manually trigger cache refresh"""
    try:
        updated = update_snapshot()
        if updated:
            return jsonify({"message": "Cache refreshed successfully"})
        else:
            return jsonify({"error": "Update failed"}), 500
    except Exception as e:
        logger.error(f"Manual refresh failed: {e}")
        return jsonify({"error": str(e)}), 500

"""-------------------------------- BACKGROUND CACHE WARMING --------------------------------"""

def refresh_cache_background():
    """
    Continuously refresh cache every 60 seconds (SNAPSHOT)
    """
    time.sleep(5) # Initial wait
    while True:
        try:
           update_snapshot()
        except Exception as e:
            logger.error(f"BACKGROUND: Cache refresh failed: {e}")
        
        time.sleep(CACHE_DURATION)

def warm_cache_on_startup():
    """
    Warm up cache immediately when server starts.
    """
    time.sleep(2)
    try:
        logger.info("STARTUP: Warming cache...")
        success = update_snapshot()
        if success:
            logger.info("STARTUP: Cache warming completed!")
        else:
            logger.info("STARTUP: Cache warming failed.")
    except Exception as e:
        logger.error(f"STARTUP: Cache warming failed: {e}")

@app.route('/home', methods=['GET'])
def basic_route():
    return "backend started successfully"

# UPDATED MAIN BLOCK
if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("Starting AMD Backend Server")
    logger.info("=" * 80)

    logger.info(f"Starting background snapshot refresh (every {CACHE_DURATION}s)...")
    background_thread = Thread(target=refresh_cache_background, daemon=True)
    background_thread.start()

    logger.info("Starting startup cache warming...")
    startup_thread = Thread(target=warm_cache_on_startup, daemon=True)
    startup_thread.start()

    logger.info("=" * 80)
    logger.info("Server ready!")
    logger.info("=" * 80)

    app.run(host='0.0.0.0', port=8088, debug=True)
