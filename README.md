# ForgeAI — Predictive Industrial Maintenance

A dark industrial AI dashboard for the AI4I 2020 machine-failure prediction project.

## Stack
- FastAPI backend
- scikit-learn Random Forest classifier (the model wired into `main.py`)
- HTML, CSS, and JavaScript frontend
- AI4I 2020 dataset
- Original notebook retained as `Predictive_Industrial_Maintenance.ipynb`

## Project files
- `main.py` — trains the Random Forest pipeline and exposes the API
- `index.html` — ForgeAI dashboard UI
- `style.css` — responsive dark industrial styling
- `ai4i2020.csv` — training dataset
- `requirements.txt` — Python dependencies
- `Predictive_Industrial_Maintenance.ipynb` — original research notebook

## Run locally
Use Python 3.10+.

1. Extract this ZIP.
2. Open a terminal in the extracted `ForgeAI_Predictive_Maintenance` folder.
3. Create and activate a virtual environment (recommended):

   Windows:
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
6. Open http://127.0.0.1:8000

FastAPI docs are available at http://127.0.0.1:8000/docs.

## API endpoints
- `GET /api/health` — backend/model status
- `GET /api/metrics` — held-out test metrics
- `POST /api/predict` — predict machine failure from sensor readings

## Notes
- The model trains on application startup; the first start may take a few seconds.
- The frontend uses the existing API contracts in `main.py`.
- The model output is an estimate, not a guarantee of real-world machine condition. Validate on representative operational data before using it for maintenance decisions.
- This is an educational prototype, not a safety-certified industrial monitoring system.
