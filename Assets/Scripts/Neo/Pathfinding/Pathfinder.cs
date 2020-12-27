using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Pathfinder : MonoBehaviour
{
    public Grid WorldGrid;

    private int DiagonalWeight = 14;
    private int StraightWeight = 10;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public List<Node> FindPath(Vector3 src, Vector3 dst)
    {
        /**
         * TODO: Return path stack
         */
        Debug.Log($"{name} Pathfinder.FindPath(src: {src}, dst: {dst})");
        Debug.Log($"{name} WorldGrid: {WorldGrid}");
        Node startNode = WorldGrid.NodeFromWorldPoint(src);
        Node goalNode = WorldGrid.NodeFromWorldPoint(dst);

        List<Node> openSet = new List<Node>();  // PriorityQueue
        HashSet<Node> closedSet = new HashSet<Node>();

        openSet.Add(startNode);

        while (openSet.Count > 0)
        {
            Node currentNode = openSet[0];
            for (int i = 1; i < openSet.Count; i++)
            {
                if (openSet[i].F_Cost < currentNode.F_Cost
                    || (openSet[i].F_Cost == currentNode.F_Cost && openSet[i].H_Cost < currentNode.H_Cost))
                {
                    currentNode = openSet[i];
                }
            }

            openSet.Remove(currentNode);
            closedSet.Add(currentNode);

            if (currentNode == goalNode)
            {
                return RetracePath(startNode, goalNode);
            }

            List<Node> neighbours = WorldGrid.GetNeighbours(currentNode);
            for (int i = 0; i < neighbours.Count; i++)
            {
                Node neighbour = neighbours[i];
                if (!neighbour.IsWalkable || closedSet.Contains(neighbour))
                {
                    continue;
                }

                int newMovementCostToNeighbour = currentNode.G_Cost + GetDistance(currentNode, neighbour);
                if (newMovementCostToNeighbour < neighbour.G_Cost
                    || !openSet.Contains(neighbour))
                {
                    neighbour.G_Cost = newMovementCostToNeighbour;
                    neighbour.H_Cost = GetDistance(neighbour, goalNode);
                    neighbour.Parent = currentNode;

                    if (!openSet.Contains(neighbour))
                    {
                        openSet.Add(neighbour);
                    }
                }
            }
        }

        return null;
    }

    private List<Node> RetracePath(Node startNode, Node endNode)
    {
        List<Node> path = new List<Node>();
        Node currentNode = endNode;

        while (currentNode != startNode)
        {
            currentNode.IsPath = true;
            path.Add(currentNode);
            currentNode = currentNode.Parent;
        }
        path.Reverse();

        return path;
    }

    public int GetDistance(Node a, Node b)
    {
        int x = Mathf.Abs(a.GridX - b.GridX);
        int y = Mathf.Abs(a.GridY - b.GridY);

        if (x > y)
        {
            return DiagonalWeight * y + StraightWeight * (x - y);
        }
        else
        {
            return DiagonalWeight * x + StraightWeight * (y - x);
        }
    }
}
