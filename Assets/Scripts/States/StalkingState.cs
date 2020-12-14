using UnityEngine;

public class StalkingState : BaseState
{
    public readonly float safeDistance = 28f;

    public override void EnterState(WarshipControllerFSM warship)
    {
        warship.m_Warship.SetEngineLevel(Warship.EngineLevel.FORWARD_MAX);
    }

    public override void Update(WarshipControllerFSM warship)
    {
        // #1. Next target point
        Vector3 currentPosition = warship.m_Warship.m_Transform.position;
        Vector3 opponentPosition = warship.m_Opponent.m_Transform.position;
        Vector3 vector = currentPosition - opponentPosition;
        float gradient = vector.z / vector.x;
        float x = Mathf.Sqrt(Mathf.Pow(safeDistance, 2) / (Mathf.Pow(gradient, 2) + 1));
        float z = gradient * x;

        float distance1 = Geometry.GetDistance(currentPosition, opponentPosition + new Vector3(x, 0f, z));
        float distance2 = Geometry.GetDistance(currentPosition, opponentPosition - new Vector3(x, 0f, z));
        Vector3 targetPosition = Vector3.zero;
        if (distance1 < distance2)
        {
            targetPosition = opponentPosition + new Vector3(x, 0f, z);
        }
        else
        {
            targetPosition = opponentPosition - new Vector3(x, 0f, z);
        }

        if (Mathf.Abs(targetPosition.x) > 90)
        {
            targetPosition.x = Mathf.Sign(targetPosition.x) * 90;
        }
        if (Mathf.Abs(targetPosition.z) > 90)
        {
            targetPosition.z = Mathf.Sign(targetPosition.z) * 90;
        }

        Vector3 targetDirection = targetPosition - currentPosition;
        Debug.DrawRay(currentPosition, targetDirection, Color.red);

        // #2. Direction
        Vector3 rotation = warship.m_Warship.m_Transform.rotation.eulerAngles;
        float angle = (Geometry.GetAngleBetween(currentPosition, targetPosition) + 360) % 360;

        float gap = angle - rotation.y;
        if ((gap > 0f && gap < 180f) || gap < -180f)
        {
            warship.m_Warship.SetRudderLevel(Warship.RudderLevel.RIGHT_MAX);
        }
        else
        {
            warship.m_Warship.SetRudderLevel(Warship.RudderLevel.LEFT_MAX);
        }

        if (Mathf.Abs(gap) > 90f)
        {
            warship.m_Warship.SetEngineLevel(Warship.EngineLevel.BACKWARD_MAX);
        }
        else
        {
            warship.m_Warship.SetEngineLevel(Warship.EngineLevel.FORWARD_MAX);
        }

        warship.m_Warship.Fire();
    }

    public override void OnCollisionEnter(WarshipControllerFSM warship)
    {

    }
}
