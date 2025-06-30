sequenceDiagram
    participant Alice as Alice (Photon Source)
    participant Eve as Eve (Interceptor / Measurement Device)
    participant Bob as Bob (Receiver)

    Alice->>Eve: Send photon with polarization θ
    Note right of Eve: Eve intercepts photon
    Eve->>Eve: Measure photon’s polarization
    Note right of Eve: Measurement collapses state → new polarization θ′
    Eve->>Bob: Resend photon with polarization θ′
    Bob->>Bob: Measure received photon
    Note over Bob: Possible mismatch revealed (θ′ ≠ θ)