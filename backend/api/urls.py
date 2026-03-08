from django.urls import path
from .views import backtest_api, event_api

urlpatterns = [
    path("backtest/", backtest_api),
    path("event-analysis/", event_api)
]