
<img width="712" height="386" alt="image" src="https://github.com/user-attachments/assets/e23b9c8c-3193-419b-bf77-314e0862ff87" />



flowchart TD
    A[Load Meeting State] --> B[Orient Participants]
    B --> C[Moderator: Select Agenda]
    C --> D[Analyze Problem]
    D --> E[Generate Proposals]
    E --> F[Critical Review]
    F --> G[Moderator: Evaluate Progress]
    G --> H{Next Action?}
    H -->|More discussion| C
    H -->|Need user input| I[Human Interrupt]
    I --> J[Process Feedback]
    J --> C
    H -->|Decision possible| K[Record Decision]
    K --> L{More agenda items?}
    L -->|Yes| C
    L -->|No| M[Finalize Meeting]
