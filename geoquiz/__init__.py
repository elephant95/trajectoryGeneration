"""geoquiz — automated geo-quiz video generator (phases 0 and 1).

Offline pipeline: ingest open data -> normalise into a GeoParquet feature
store -> resolve a region -> build a scene -> render a Reel.
"""

__version__ = "0.1.0"


def __getattr__(name):          # lazy: importing the package stays cheap
    if name == "GeoQuiz":
        from .api import GeoQuiz
        return GeoQuiz
    raise AttributeError(name)
