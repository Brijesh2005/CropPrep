# STAM — Sequence Diagram

## Online observation build

```
 Frontend/Client   STAM facade     Matcher        Dataset Manager     Caches
      │               │              │                  │                │
      │ build_observation(lon,lat)   │                  │                │
      │──────────────►│              │                  │                │
      │               │ cache lookup │                  │                │
      │               │──────────────│──────────────────│───────────────►│
      │               │              │                  │      MISS      │
      │               │ location_info│                  │                │
      │               │──────────────│────────► query metadata / load    │
      │               │              │                  │                │
      │               │              │◄────── locations + boundaries     │
      │               │              │                  │                │
      │               │ resolve_temporal               │                │
      │               │──────────────│                  │                │
      │               │ match_tabular│                  │                │
      │               │──────────────│────────► list_csvs + load_csv     │
      │               │              │                  │                │
      │               │ match_images │                  │                │
      │               │──────────────│────────► query_metadata(NDVI/EVI) │
      │               │              │                  │                │
      │               │ build sequence (pair/sort/gaps)│                │
      │               │──────────────│                  │                │
      │               │ quality pass │                  │                │
      │               │──────────────│                  │                │
      │               │ cache store  │                  │                │
      │               │──────────────│──────────────────│───────────────►│
      │  observation  │              │                  │                │
      │◄──────────────│              │                  │                │
```

## Patch extraction

```
 Caller        STAM facade      PatchGenerator      Dataset Manager
   │  get_patch(path,lon,lat)       │                     │
   │───────────────►│               │                     │
   │                │  image_metadata(path)               │
   │                │──────────────│────────────────────►│  (cached record)
   │                │              │◄────────────────────│
   │                │  point→pixel + window + read       │
   │                │──────────────│────────────────────►│  load_image(window)
   │                │              │◄────────────────────│
   │  RasterPatch   │              │                     │
   │◄───────────────│              │                     │
```

## Key timing note

`initialize()` performs the only "expensive" setup: loading boundaries and
building the spatial/boundary indexes (one-time, cached). Every
`build_observation` afterwards is metadata lookups + windowed reads only.
