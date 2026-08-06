# R2.2 Extended Metadata Repository

```mermaid
flowchart LR
    subgraph Writers["writers"]
        REG["ProviderRegistry → save_provider"]
        SPIDX["SpatialIndexImpl → save_spatial_many"]
        HCB["HistoricalContextBuilder → save_temporal"]
        PE["PatchExtractor → save_patch"]
    end

    subgraph DB["metadata.db (shared, WAL)"]
        R1["metadata_records (R1.2)"]
        P["provider_metadata<br/>(name) PK"]
        S["spatial_metadata<br/>(name, kind) PK<br/>idx kind · district"]
        T["temporal_metadata<br/>(index_type, year, resolution) PK"]
        PT["patch_metadata<br/>(patch_id) PK"]
    end

    subgraph Readers["readers"]
        PR["list_providers · provider_count"]
        SR["list_spatial · spatial_count"]
        TR["list_temporal(index/year/res)"]
        PTR["list_patches · patch_count"]
    end

    REG --> P
    SPIDX --> S
    HCB --> T
    PE --> PT
    R1 -. shares _db.py connection discipline .- P
    P --> PR
    S --> SR
    T --> TR
    PT --> PTR

    style DB fill:#fff3e0
    style Writers fill:#e8f5e9
    style Readers fill:#e3f2fd
```
