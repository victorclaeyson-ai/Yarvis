"""The weather tool, against Open-Meteo's documented answers."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx2
import pytest

from yarvis.tools.base import Tier, ToolError
from yarvis.tools.weather import FORECAST_URL, GEOCODING_URL, Weather, WeatherArgs

TODAY = date(2026, 11, 15)
LEUVEN = {
    "name": "Leuven", "latitude": 50.87959, "longitude": 4.70093, "country_code": "BE",
    "country": "Belgium", "admin1": "Flanders", "admin2": "Flemish Brabant",
}  # fmt: skip
LEUVEN_USA = {**LEUVEN, "latitude": 40.0, "longitude": -90.0, "country": "United States",
              "country_code": "US", "admin1": "Ohio"}  # fmt: skip
FORECAST: dict[str, Any] = {
    "timezone": "Europe/Brussels",
    "current": {
        "time": "2026-11-15T17:45", "temperature_2m": 7.3, "apparent_temperature": 4.1,
        "weather_code": 61, "wind_speed_10m": 18.4,
    },
    "daily": {
        "time": [f"2026-11-{15 + n}" for n in range(7)],
        "weather_code": [61, 3, 0, 95, 2, 71, 45],
        "temperature_2m_max": [9.1, 11.6, 12.0, 10.0, 8.0, 2.0, 5.0],
        "temperature_2m_min": [4.2, 5.4, 3.0, 6.0, 1.0, -2.4, 0.0],
        "precipitation_probability_max": [80, 10, 0, 90, 5, 60, 20],
        "precipitation_sum": [3.2, 0.0, 0.0, 12.5, 0.0, 2.0, 0.0],
        "sunrise": [f"2026-11-{15 + n}T07:58" for n in range(7)],
        "sunset": [f"2026-11-{15 + n}T16:52" for n in range(7)],
    },
}  # fmt: skip


class OpenMeteo:
    def __init__(self, places: list[dict[str, Any]] | None = None, status: int = 200) -> None:
        self.places = [LEUVEN] if places is None else places
        self.status = status
        self.requests: list[httpx2.Request] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        url = str(request.url).split("?")[0]
        if url == GEOCODING_URL:
            body: dict[str, Any] = {"results": self.places} if self.places else {}
            return httpx2.Response(self.status, json=body)
        assert url == FORECAST_URL
        return httpx2.Response(self.status, json=FORECAST)


def weather(service: OpenMeteo, home: str | None = "Leuven, Belgium") -> Weather:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(service.handler))
    return Weather(home, http=http, today=lambda: TODAY)


async def test_now_and_today_at_home() -> None:
    service = OpenMeteo()
    tool = weather(service)
    said = await tool.run(WeatherArgs())
    assert said == (
        "Leuven, Belgium. Now: 7°C, feels like 4°C, light rain, wind 18 km/h. Today: light "
        "rain, 4°C to 9°C, 80% chance of rain (3.2 mm). Sunrise 07:58, sunset 16:52."
    )
    geocoding, forecast = service.requests
    assert geocoding.url.params["name"] == "Leuven"
    assert forecast.url.params["latitude"] == "50.8796"
    assert forecast.url.params["timezone"] == "auto"
    assert tool.tier is Tier.READ and tool.untrusted_output
    assert tool.describe(WeatherArgs()) == "check the weather in your area"


async def test_another_day_and_place() -> None:
    service = OpenMeteo(places=[LEUVEN_USA, LEUVEN])
    tool = weather(service)
    said = await tool.run(WeatherArgs(place="Leuven, Belgium", days_ahead=1))
    assert said.startswith("Leuven, Belgium. Tomorrow: overcast, 5°C to 12°C, 10% chance of rain.")
    assert "Now:" not in said
    said = await tool.run(WeatherArgs(place="Leuven, Belgium", days_ahead=5))
    assert "On 2026-11-20: light snow, -2°C to 2°C, 60% chance of rain (2.0 mm)." in said
    assert len(service.requests) == 3  # the place was looked up once
    args = WeatherArgs(place="Ghent", days_ahead=1)
    assert weather(OpenMeteo()).describe(args) == "check the weather in Ghent for tomorrow"
    assert weather(OpenMeteo()).describe(WeatherArgs(place="Ghent", days_ahead=3)) == (
        "check the weather in Ghent in 3 days"
    )


@pytest.mark.parametrize(
    ("service", "home", "args", "message"),
    [
        (OpenMeteo(), None, WeatherArgs(), "No place was given"),
        (OpenMeteo(places=[]), "Leuven", WeatherArgs(place="Atlantis"), "No place called"),
        (OpenMeteo(status=500), "Leuven", WeatherArgs(), "could not be reached"),
        (OpenMeteo(places=[{"name": "Odd"}]), "Leuven", WeatherArgs(), "could not be reached"),
    ],
)
async def test_what_can_go_wrong(
    service: OpenMeteo, home: str | None, args: WeatherArgs, message: str
) -> None:
    with pytest.raises(ToolError, match=message):
        await weather(service, home).run(args)


async def test_offline_or_odd_answers() -> None:
    def offline(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("offline")

    tool = Weather("Leuven", http=httpx2.AsyncClient(transport=httpx2.MockTransport(offline)))
    with pytest.raises(ToolError, match="could not be reached"):
        await tool.run(WeatherArgs())

    def odd(request: httpx2.Request) -> httpx2.Response:
        if str(request.url).startswith(GEOCODING_URL):
            return httpx2.Response(200, json={"results": [LEUVEN]})
        return httpx2.Response(200, json=["not", "an", "object"])

    tool = Weather("Leuven", http=httpx2.AsyncClient(transport=httpx2.MockTransport(odd)))
    with pytest.raises(ToolError, match="could not be reached"):
        await tool.run(WeatherArgs())

    forecasts: tuple[dict[str, object], ...] = ({"current": {}}, {"daily": {"time": []}})
    for forecast in forecasts:

        def no_days(request: httpx2.Request, body: object = forecast) -> httpx2.Response:
            if str(request.url).startswith(GEOCODING_URL):
                return httpx2.Response(200, json={"results": [LEUVEN]})
            return httpx2.Response(200, json=body)

        transport = httpx2.MockTransport(no_days)
        tool = Weather("Leuven", http=httpx2.AsyncClient(transport=transport))
        with pytest.raises(ToolError, match="could not be reached"):
            await tool.run(WeatherArgs())


async def test_a_forecast_from_another_day_still_answers() -> None:
    """The computer's date and the forecast's may differ around midnight."""
    service = OpenMeteo()
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(service.handler))
    tool = Weather("Leuven", http=http, today=lambda: date(2030, 1, 1))
    said = await tool.run(WeatherArgs(days_ahead=1))
    assert "Tomorrow: overcast" in said  # the second day of the forecast
