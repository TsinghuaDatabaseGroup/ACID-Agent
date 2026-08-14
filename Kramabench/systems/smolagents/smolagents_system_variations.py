from .deepresearch import SmolagentsDeepResearch
from .reflexion import SmolagentsReflexion
from .pdt import SmolagentsPDT

class SmolagentsDeepResearchClaude37Sonnet(SmolagentsDeepResearch):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="claude-3-7-sonnet-latest",
            name="SmolagentsDeepResearchClaude37Sonnet",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )

class SmolagentsDeepResearchClaude45Sonnet(SmolagentsDeepResearch):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="claude-sonnet-4-5-20250929",
            name="SmolagentsDeepResearchClaude45Sonnet",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )

class SmolagentsDeepResearchGPTo3(SmolagentsDeepResearch):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="o3-2025-04-16",
            name="SmolagentsDeepResearchGPTo3",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )

class SmolagentsReflexionClaude37Sonnet(SmolagentsReflexion):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="claude-3-7-sonnet-latest",
            name="SmolagentsReflexionClaude37Sonnet",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )

class SmolagentsReflexionClaude45Sonnet(SmolagentsReflexion):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="claude-sonnet-4-5-20250929",
            name="SmolagentsReflexionClaude45Sonnet",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )

class SmolagentsReflexionGPTo3(SmolagentsReflexion):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="o3-2025-04-16",
            name="SmolagentsReflexionGPTo3",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )

class SmolagentsPDTClaude37Sonnet(SmolagentsPDT):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="claude-3-7-sonnet-latest",
            name="SmolagentsPDTClaude37Sonnet",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )
        
class SmolagentsPDTClaude45Sonnet(SmolagentsPDT):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="claude-sonnet-4-5-20250929",
            name="SmolagentsPDTClaude45Sonnet",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )

class SmolagentsPDTGPTo3(SmolagentsPDT):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(
            model="o3-2025-04-16",
            name="SmolagentsPDTGPTo3",
            verbose=verbose,
            supply_data_snippet=True,
            *args, **kwargs
        )