"""The weather, from Open-Meteo: free, no account, no API key.

Asking a weather service directly is faster and more reliable than a web search: one
request for the place (geocoding, cached) and one for the forecast, both over HTTPS. Only
the place name and its coordinates leave the computer.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Final, cast

import httpx2
from pydantic import Field

from yarvis.tools.base import Arguments, Tier, Tool, ToolError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

GEOCODING_URL: Final = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL: Final = "https://api.open-meteo.com/v1/forecast"
_DAYS: Final = 7
_UNREACHABLE: Final = "The weather service could not be reached. Check the internet connection."

# WMO weather interpretation codes, as Open-Meteo reports them.
_CONDITIONS: Final[dict[int, str]] = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with heavy hail",
}  # fmt: skip


class WeatherArgs(Arguments):
    """Arguments of ``weather``."""

    place: str = Field(
        default="",
        max_length=80,
        description="City, optionally with its country ('Ghent, Belgium'). Empty: home.",
    )
    days_ahead: int = Field(default=0, ge=0, le=_DAYS - 1, description="0 = today.")


class Weather(Tool[WeatherArgs]):
    """Current weather and the forecast for up to a week."""

    name = "weather"
    description = (
        "The weather: now and today (days_ahead 0) or a day in the coming week, for a "
        "place or (empty place) the user's home."
    )
    tier = Tier.READ
    Args = WeatherArgs
    untrusted_output = True  # place names come from an outside database

    def __init__(
        self,
        home: str | None,
        *,
        http: httpx2.AsyncClient | None = None,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._home = home
        self._http = http or httpx2.AsyncClient(timeout=8.0, follow_redirects=False)
        self._today = today
        self._places: dict[str, tuple[str, float, float]] = {}

    async def run(self, args: WeatherArgs) -> str:
        """Look it up."""
        place = args.place.strip() or self._home
        if not place:
            msg = "No place was given and the user has not set a home location."
            raise ToolError(msg)
        name, latitude, longitude = await self._find(place)
        forecast = await self._get(
            FORECAST_URL,
            {
                "latitude": f"{latitude:.4f}",
                "longitude": f"{longitude:.4f}",
                "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_probability_max,precipitation_sum,sunrise,sunset"
                ),
                "timezone": "auto",
                "forecast_days": str(_DAYS),
            },
        )
        return _describe(name, forecast, args.days_ahead, self._today())

    def describe(self, args: WeatherArgs) -> str:
        """'check the weather in Ghent'."""
        where = args.place.strip() or "your area"
        when = {0: "", 1: " for tomorrow"}.get(args.days_ahead, f" in {args.days_ahead} days")
        return f"check the weather in {where}{when}"

    async def _find(self, place: str) -> tuple[str, float, float]:
        key = place.casefold()
        if key in self._places:
            return self._places[key]
        town, _, region = (part.strip() for part in place.partition(","))
        found = _list(
            (await self._get(GEOCODING_URL, {"name": town, "count": "10", "language": "en"})).get(
                "results"
            )
        )
        if region:  # "Ghent, Belgium": prefer the one in that country or region
            wanted = region.casefold()
            found = [r for r in found if _in_region(r, wanted)] or found
        if not found:
            msg = f"No place called {place!r} was found."
            raise ToolError(msg)
        best = found[0]
        country = str(best.get("country", ""))
        name = f"{best.get('name', town)}, {country}" if country else str(best.get("name", town))
        try:
            result = (name[:80], float(best["latitude"]), float(best["longitude"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError(_UNREACHABLE) from exc
        self._places[key] = result
        return result

    async def _get(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        try:
            response = await self._http.get(url, params=params)
        except httpx2.HTTPError as exc:
            raise ToolError(_UNREACHABLE) from exc
        if response.status_code != 200:  # noqa: PLR2004
            logger.warning("Open-Meteo answered HTTP %s", response.status_code)
            raise ToolError(_UNREACHABLE)
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise ToolError(_UNREACHABLE)
        return cast("dict[str, Any]", payload)


def _in_region(result: dict[str, Any], wanted: str) -> bool:
    fields = ("country", "country_code", "admin1", "admin2")
    return any(str(result.get(field, "")).casefold().startswith(wanted) for field in fields)


def _list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        cast("dict[str, Any]", item)
        for item in cast("list[object]", value)
        if isinstance(item, dict)
    ]


def _describe(name: str, forecast: dict[str, Any], days_ahead: int, today: date) -> str:
    daily = forecast.get("daily")
    if not isinstance(daily, dict):
        raise ToolError(_UNREACHABLE)
    days = cast("dict[str, list[Any]]", daily)
    try:
        index = days["time"].index((today + timedelta(days=days_ahead)).isoformat())
    except (KeyError, ValueError):
        index = min(days_ahead, len(days.get("time", [])) - 1)
    if index < 0:
        raise ToolError(_UNREACHABLE)

    def day(field: str) -> Any:  # noqa: ANN401 - numbers or strings from JSON
        values = days.get(field) or []
        return values[index] if index < len(values) else None

    label = {0: "Today", 1: "Tomorrow"}.get(days_ahead, f"On {day('time')}")
    parts = [f"{name}."]
    current = forecast.get("current")
    if days_ahead == 0 and isinstance(current, dict):
        now = cast("dict[str, Any]", current)
        parts.append(
            f"Now: {_degrees(now.get('temperature_2m'))}, feels like "
            f"{_degrees(now.get('apparent_temperature'))}, {_condition(now.get('weather_code'))}, "
            f"wind {_number(now.get('wind_speed_10m'))} km/h."
        )
    rain = f"{_number(day('precipitation_probability_max'))}% chance of rain"
    amount = day("precipitation_sum")
    if isinstance(amount, int | float) and amount > 0:
        rain += f" ({amount:.1f} mm)"
    parts.append(
        f"{label}: {_condition(day('weather_code'))}, {_degrees(day('temperature_2m_min'))} to "
        f"{_degrees(day('temperature_2m_max'))}, {rain}. Sunrise {_clock(day('sunrise'))}, "
        f"sunset {_clock(day('sunset'))}."
    )
    return " ".join(parts)


def _condition(code: object) -> str:
    return _CONDITIONS.get(code, "unknown conditions") if isinstance(code, int) else "unknown"


def _degrees(value: object) -> str:
    return f"{round(value)}°C" if isinstance(value, int | float) else "?"


def _number(value: object) -> str:
    return str(round(value)) if isinstance(value, int | float) else "?"


def _clock(value: object) -> str:
    return value.rpartition("T")[2] if isinstance(value, str) else "?"
