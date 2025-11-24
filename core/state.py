import json
import logging
from firebase_admin import initialize_app, firestore, credentials

logging.basicConfig(level=logging.INFO)

# --- Global Firebase Variables (Provided by Runtime) ---
# NOTE: These variables are expected to be available in the execution environment.
# We parse them from the environment.
APP_ID = os.environ.get('__app_id', 'default-app-id')
FIREBASE_CONFIG_JSON = os.environ.get('__firebase_config', '{}')

# --- Firebase Initialization ---
db = None
try:
    if FIREBASE_CONFIG_JSON:
        # 1. Prepare credentials using the config provided by the environment
        firebase_config = json.loads(FIREBASE_CONFIG_JSON)
        
        # We need to construct a credential object that can be initialized globally
        # If running in a secure environment where firebase_admin is used, the 
        # config often contains the keys needed for service account initialization.
        # For this controlled environment, we rely on the runtime handling the credentials,
        # but we must initialize the app context.
        
        # A simple, secure way to handle initialization:
        try:
            # Check if an app is already initialized to prevent errors
            initialize_app(options=firebase_config)
        except ValueError:
            # If the app is already initialized, just continue (e.g., if running multiple times)
            pass 
        
        db = firestore.client()
        logging.info("✅ Firestore client initialized successfully.")
    else:
        logging.error("❌ __firebase_config environment variable is missing or empty.")
except Exception as e:
    logging.error(f"❌ Failed to initialize Firebase: {e}")
    db = None


def save_lp_state(bot_id: int, state_updates: dict):
    """
    Saves or updates the bot's state dictionary to Firestore.
    
    The state is saved in the public shared space so that the portfolio 
    API can easily read it.
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
        # This allows other functions (like wallet balance reporters) to update 
        # other fields without overwriting this data.
        doc_ref.set(state_updates, merge=True)
        
        logging.info(f"💾 Successfully saved LP state for Bot {bot_id} to Firestore.")

    except Exception as e:
        logging.error(f"❌ Error saving state for Bot {bot_id}: {e}")
        logging.debug(f"State attempted to save: {state_updates}")

# NOTE: The old 'update_lp_state' is now replaced by 'save_lp_state' which handles 
# dictionary inputs, aligning with the new central state structure.
