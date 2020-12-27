using UnityEngine;

public class Node
{
    public bool IsWalkable;
    public Vector3 WorldPosition;
    public Node Parent;
    public bool IsPath;

    public int GridX;
    public int GridY;

    public int G_Cost;  // Distance from the source
    public int H_Cost;  // Distance to the goal

    public int F_Cost
    {
        get => G_Cost + H_Cost;
    }

    public Node(bool isWalkable, Vector3 worldPosition, int gridX, int gridY)
    {
        IsWalkable = isWalkable;
        WorldPosition = worldPosition;
        GridX = gridX;
        GridY = gridY;
    }
}
