---
config:
  look: handDrawn
---
flowchart LR
  subgraph Alice
    direction TB
    A1["State preparation"]
    A2["Information reconciliation"]
    A3["Parameter estimation"]
    A4["Privacy amplification"]
    A5["Secure data application"]
  end
  subgraph Eve
    direction TB
    Q["Quantum<br/>channel"]:::qc
    E["Authenticated<br/>channel"]:::ac
    C["Communication<br/>channel"]:::cc
  end
  subgraph Bob
    direction TB
    B1["State measurement"]
    B2["Information reconciliation"]
    B3["Parameter estimation"]
    B4["Privacy amplification"]
    B5["Secure data application"]
  end
  A1 -->|"α̂_sig, β_LO"| Q
  Q --> B1
  E --> |mapping, syndromes, hash| A2
  A2 -->|correctness check| E
  E <--> B2
  A3 -->|erroneous frames, calibration data| E
  E --> B3
  A4 -->|seed, secret key length| E
  E --> B4
  A5 -->|ciphertext A → B| C
  C --> |ciphertext A → B| B5
  B5 -->|ciphertext B → A| C
  C --> |ciphertext B → A| A5
  classDef qc fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px;
  classDef ac fill:#E3F2FD,stroke:#2196F3,stroke-width:2px,stroke-dasharray:5,5;
  classDef cc fill:#F5F5F5,stroke:#9E9E9E,stroke-width:2px,stroke-dasharray:5,5;