using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WarshipControllerFSM : MonoBehaviour
{
    public int m_PlayerId;
    public Warship m_Warship;
    public Warship m_Opponent;

    private BaseState m_CurrentState;

    public readonly IdleState m_IdleState = new IdleState();
    public readonly StalkingState m_StakingState = new StalkingState();
    public readonly CombatState m_CombatState = new CombatState();

    // Start is called before the first frame update
    void Start()
    {
        TransitionToState(m_IdleState);
    }

    // Update is called once per frame
    void Update()
    {
        // Debug.Log($"WarshipControllerFSM -> Update {m_CurrentState}");
        m_CurrentState.Update(this);
    }

    public void TransitionToState(BaseState state)
    {
        m_CurrentState = state;
        m_CurrentState.EnterState(this);
    }

    private void OnCollisionEnter(Collision collision)
    {
        m_CurrentState.OnCollisionEnter(this);
    }
}
