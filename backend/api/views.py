import json
from django.http import JsonResponse
from .services.backtester import run_backtest
from .services.risk_model import classify_risk
from .services.database import save_result
from .services.event_analysis import analyze_event

def backtest_api(request):

    if request.method != "POST":
        return JsonResponse({"error": "POST only"})

    data = json.loads(request.body)

    symbol = data["symbol"]
    strategy = data["strategy"]

    result = run_backtest(symbol, strategy)

    risk = classify_risk(result["metrics"])

    result["risk"] = risk

    save_result(result)

    return JsonResponse(result)


def event_api(request):

    data = json.loads(request.body)

    symbol = data["symbol"]

    analysis = analyze_event(symbol)

    return JsonResponse(analysis)