import os
import json
import logging
from firebase_admin import initialize_app, firestore, credentials

# Set up logging for visibility during execution
logging.basicConfig(level=logging.INFO)

# --- Global Firebase Variables (Provided by Runtime) ---
# NOTE: These variables are read from the execution environment.
APP_ID = os.environ.get('__app_id', 'default-app-id')
FIREBASE_CONFIG_JSON = os.environ.get('__firebase_config', '{}')

# --- Firebase Initialization ---
db = None
try:
    if FIREBASE_CONFIG_JSON:
        # 1. Parse the configuration JSON provided by the environment
        firebase_config = json.loads(FIREBASE_CONFIG_JSON)
        
        # 2. Create service account credentials from the config dict. 
        # This is required by the Python firebase_admin SDK.
        cred = credentials.Certificate(firebase_config)
        
        # 3. Initialize the app. We wrap this in a try/except to handle the 
        # ValueError that occurs if the function is called multiple times 
        # (e.g., in a testing or restart scenario).
        try:
            initialize_app(cred)
        except ValueError:
            # This is expected if the app is already initialized. Continue.
            pass
        
        # 4. Get the Firestore client instance
        db = firestore.client()
        logging.info("✅ Firestore client initialized successfully.")
    else:
        logging.error("❌ __firebase_config environment variable is missing or empty.")
except Exception as e:
    # Catch any critical failure during initialization (e.g., bad JSON, network)
    logging.error(f"❌ CRITICAL: Failed to initialize Firebase: {e}")
    db = None


def save_lp_state(bot_id: int, state_updates: dict):
    """
    Saves or updates the bot's LP state dictionary to the public Firestore path.
    This data is used by the external API to report the bot's portfolio status.
    Path: /artifacts/{__app_id}/public/data/bot_states/{bot_id}

    Args:
        bot_id: The unique ID of the bot/user.
        state_updates: A dictionary containing the LP data (price, active, lp_usdt, etc.)
                       to be merged into the bot's main state document.
    """
    if db is None:
        logging.error(f"❌ Cannot save state for Bot {bot_id}: Firestore not initialized.")
        return

    try:
        # Define the document path for the bot's state in the public collection
        doc_path = f"artifacts/{APP_ID}/public/data/bot_states/{bot_id}"
        doc_ref = db.document(doc_path)

        # Merge the incoming LP data into the main document. 
        # This is safe and prevents overwriting other state fields.
        doc_ref.set(state_updates, merge=True)
        
        logging.info(f"💾 Successfully saved LP state for Bot {bot_id} to Firestore.")

    except Exception as e:
        logging.error(f"❌ Error saving state for Bot {bot_id}: {e}")
