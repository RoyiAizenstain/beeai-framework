import os
import sys
import ollama
from langfuse import Langfuse
# שורות הייבוא הבעייתיות (כמו UpdateTrace) הוסרו

# ======================================================================
# --- 1. הגדרת המפתחות כמשתני סביבה (הדרך המועדפת על Langfuse) ---
# אנו מגדירים את המפתחות ישירות לתוך os.environ כדי שהאתחול יעבוד
# ודא שהמפתחות האלה נכונים!
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-c28c5169-eb9a-403a-9169-b4f2e7522e4d"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-9effb0e8-28d4-4848-9eb7-9f4c0d52b3af"
os.environ["LANGFUSE_BASE_URL"] = "http://localhost:3000"

# --- אתחול Langfuse ---
# הקליינט יאתחל את עצמו אוטומטית באמצעות משתני הסביבה שהוגדרו
try:
    langfuse = Langfuse()
    print("Langfuse client initialized successfully from environment variables.")
except Exception as e:
    print(f"🚨🚨 FATAL: Langfuse initialization failed. Error: {e}")
    sys.exit(1) # יציאה אם האתחול נכשל

# ======================================================================

# --- 2. פרמטרים לקריאת Ollama ו-Evaluation ---
OLLAMA_MODEL = "llama3.1:8b" 

# רשימת שאלות לבדיקה (test cases)
TEST_CASES = [
    {
        "prompt": "Write a short, fun fact about the BeeAI framework, no more than 10 words.",
        "expected_keywords": ["BeeAI", "framework"],  # מילות מפתח שצריכות להופיע
        "max_length": 100,  # אורך מקסימלי
    },
    {
        "prompt": "What is 2+2?",
        "expected_answer": "4",  # תשובה מדויקת
    },
    {
        "prompt": "Say hello in one word.",
        "expected_keywords": ["hello"],
        "max_length": 10,
    }
]

print(f"Starting Langfuse Evaluation for model: {OLLAMA_MODEL}")
print(f"Number of test cases: {len(TEST_CASES)}\n")

# --- 3. פונקציה לבדיקת Evaluation ---
def evaluate_response(response_text, test_case):
    """בודק את התשובה לפי הקריטריונים של test case"""
    score = 0.0
    max_score = 0.0
    reasons = []
    
    # בדיקת מילות מפתח
    if "expected_keywords" in test_case:
        max_score += 1.0
        keywords_found = sum(1 for kw in test_case["expected_keywords"] 
                            if kw.lower() in response_text.lower())
        keyword_score = keywords_found / len(test_case["expected_keywords"])
        score += keyword_score
        reasons.append(f"Keywords: {keywords_found}/{len(test_case['expected_keywords'])} found")
    
    # בדיקת תשובה מדויקת
    if "expected_answer" in test_case:
        max_score += 1.0
        if test_case["expected_answer"].lower().strip() in response_text.lower():
            score += 1.0
            reasons.append("Exact answer match")
        else:
            reasons.append(f"Expected '{test_case['expected_answer']}', got different answer")
    
    # בדיקת אורך מקסימלי
    if "max_length" in test_case:
        max_score += 1.0
        if len(response_text) <= test_case["max_length"]:
            score += 1.0
            reasons.append(f"Length OK ({len(response_text)} <= {test_case['max_length']})")
        else:
            reasons.append(f"Too long ({len(response_text)} > {test_case['max_length']})")
    
    final_score = score / max_score if max_score > 0 else 0.0
    return final_score, reasons

# --- 4. ביצוע Evaluation על כל test case ---
results = []

for idx, test_case in enumerate(TEST_CASES, 1):
    print(f"\n{'='*60}")
    print(f"Test Case {idx}/{len(TEST_CASES)}")
    print(f"{'='*60}")
    print(f"Prompt: {test_case['prompt']}")
    
    try:
        # יצירת Trace/Span לכל test case
        with langfuse.start_as_current_span(
            name=f"evaluation-test-{idx}",
            metadata={"test_case": idx, "prompt": test_case['prompt']}
        ):
            # התחלת Generation
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="ollama-generation",
                model=OLLAMA_MODEL,
                input={"prompt": test_case['prompt']},
            ) as generation:
                
                # קריאת Ollama
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{'role': 'user', 'content': test_case['prompt']}],
                )
                
                # חילוץ התוצאות
                completion_text = response['message']['content']
                prompt_tokens = response.get('prompt_eval_count', 0)
                completion_tokens = response.get('eval_count', 0)
                
                # עדכון Generation
                try:
                    if hasattr(generation, 'update'):
                        generation.update(
                            output={"completion": completion_text},
                            metadata={
                                "usage": {
                                    "input_tokens": prompt_tokens, 
                                    "output_tokens": completion_tokens,
                                    "total_tokens": prompt_tokens + completion_tokens
                                }
                            }
                        )
                except:
                    pass
                
                # ביצוע Evaluation
                eval_score, eval_reasons = evaluate_response(completion_text, test_case)
                
                # יצירת Score ב-Langfuse
                langfuse.score_current_trace(
                    name="evaluation_score",
                    value=eval_score,
                    comment="; ".join(eval_reasons)
                )
                
                # שמירת התוצאות
                results.append({
                    "test_case": idx,
                    "prompt": test_case['prompt'],
                    "response": completion_text,
                    "score": eval_score,
                    "reasons": eval_reasons,
                    "tokens": prompt_tokens + completion_tokens
                })
                
                # הדפסת התוצאות
                print(f"\nResponse: {completion_text}")
                print(f"Score: {eval_score:.2%}")
                print(f"Reasons: {'; '.join(eval_reasons)}")
                
    except ollama.ResponseError as e:
        print(f"Error: Ollama connection failed - {e}")
        results.append({
            "test_case": idx,
            "prompt": test_case['prompt'],
            "response": None,
            "score": 0.0,
            "reasons": [f"Error: {str(e)}"],
            "tokens": 0
        })
    except Exception as e:
        print(f"Error: {e}")
        results.append({
            "test_case": idx,
            "prompt": test_case['prompt'],
            "response": None,
            "score": 0.0,
            "reasons": [f"Error: {str(e)}"],
            "tokens": 0
        })

# --- 5. סיכום התוצאות ---
print(f"\n{'='*60}")
print("EVALUATION SUMMARY")
print(f"{'='*60}")
total_score = sum(r['score'] for r in results)
avg_score = total_score / len(results) if results else 0.0
total_tokens = sum(r['tokens'] for r in results)

print(f"Average Score: {avg_score:.2%}")
print(f"Total Tokens Used: {total_tokens}")
print(f"\nDetailed Results:")
for r in results:
    print(f"  Test {r['test_case']}: {r['score']:.2%} - {', '.join(r['reasons'])}")

# --- 6. סיום ושליחת הנתונים לשרת Langfuse המקומי ---
langfuse.flush()
print(f"\n{'='*60}")
print("Langfuse Trace sent successfully (check http://localhost:3000)")
print(f"{'='*60}")