#!/usr/bin/env python3
"""Entry point: inicia a API FastAPI."""
import uvicorn
if __name__ == '__main__':
    uvicorn.run('esg_ml.interfaces.api.principal:app',
                host='0.0.0.0', port=8000, reload=True)
