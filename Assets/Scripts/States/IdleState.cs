using UnityEngine;

public class IdleState : BaseState
{
    public override void EnterState(WarshipControllerFSM warship)
    {
        warship.m_Warship.Reset();
    }

    public override void Update(WarshipControllerFSM warship)
    {
        float distance = Geometry.GetDistance(
            warship.m_Warship.m_Transform.position,
            warship.m_Opponent.m_Transform.position);

        if (distance < 1000f)
        {
            warship.TransitionToState(warship.m_StakingState);
        }
    }

    public override void OnCollisionEnter(WarshipControllerFSM warship)
    {

    }
}
