"""AgriScape AI: Sentinel-2 to Unreal Engine weightmap pipeline.

Modules:
    download          -- Stage 0: fetch Sentinel-2 L2A bands via STAC
    ndvi_stack        -- Stage 1-2: build multi-temporal NDVI stack
    classify          -- Stage 3: Random Forest classification
    export_weightmaps -- Stage 4: Unreal-ready per-class PNG weightmaps
"""

__version__ = "0.2.0"
