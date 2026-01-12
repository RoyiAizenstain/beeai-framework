import pickle
from pprint import pprint

# החלף את 'file.pkl' בנתיב לקובץ שלך
# path = r"python\eval\agents\requirement\agent_run_checkpoint.pkl"
path = r"python\eval\agents\requirement\eval_results_raw.pkl"


try:
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    print("--- תוכן הקובץ ---")
    pprint(data) # pprint מדפיס מבנים מורכבים בצורה קריאה
except Exception as e:
    print(f"שגיאה בטעינת הקובץ: {e}")