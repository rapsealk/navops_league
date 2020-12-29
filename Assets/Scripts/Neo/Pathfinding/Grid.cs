using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Grid : MonoBehaviour
{
    public LayerMask UnwalkableLayerMask;
    public Vector2 GridWorldSize;
    public float NodeRadius;

    private Node[,] WorldGrid;
    private float NodeDiameter;
    private Vector2Int GridSize;
    private bool IsGizmosInitialized = false;

    // Start is called before the first frame update
    void Start()
    {
        NodeDiameter = NodeRadius * 2;
        GridSize = new Vector2Int(Mathf.RoundToInt(GridWorldSize.x / NodeDiameter),
                                  Mathf.RoundToInt(GridWorldSize.y / NodeDiameter));

        CreateGrid();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnDrawGizmos()
    {
        if (!IsGizmosInitialized)   // true
        {
            Debug.Log($"{name} OnDrawGizmos");

            Gizmos.DrawWireCube(transform.position, new Vector3(GridWorldSize.x, 1f, GridWorldSize.y));

            if (WorldGrid != null)
            {
                for (uint x = 0; x < GridSize.x; x++)
                {
                    for (uint y = 0; y < GridSize.y; y++)
                    {
                        Node node = WorldGrid[x, y];
                        Gizmos.color = node.IsWalkable ? Color.white : Color.red;
                        if (node.IsPath)
                        {
                            Gizmos.color = Color.green;
                        }
                        Gizmos.DrawCube(node.WorldPosition, Vector3.one * (NodeDiameter - 0.1f));
                    }
                }
            }

            IsGizmosInitialized = true;
        }
    }

    private void CreateGrid()
    {
        Debug.Log($"{name} CreateGrid");

        WorldGrid = new Node[GridSize.x, GridSize.y];
        Vector3 worldBottomLeft = transform.position - Vector3.right * GridWorldSize.x / 2 - Vector3.forward * GridWorldSize.y / 2;
        for (int x = 0; x < GridSize.x; x++)
        {
            for (int y = 0; y < GridSize.y; y++)
            {
                Vector3 worldPoint = worldBottomLeft + Vector3.right * (x * NodeDiameter + NodeRadius) + Vector3.forward * (y * NodeDiameter + NodeRadius);
                bool walkable = !Physics.CheckSphere(worldPoint, NodeRadius, layerMask: UnwalkableLayerMask);
                WorldGrid[x, y] = new Node(walkable, worldPoint, x, y);
            }
        }
    }

    public Node NodeFromWorldPoint(Vector3 worldPosition)
    {
        Debug.Log($"{name} NodeFromWorldPoint: {worldPosition}");
        float percentX = 0.5f + worldPosition.x / GridWorldSize.x;
        float percentY = 0.5f + worldPosition.z / GridWorldSize.y;
        percentX = Mathf.Clamp01(percentX);
        percentY = Mathf.Clamp01(percentY);

        int x = Mathf.FloorToInt(Mathf.Min(GridSize.x * percentX, GridSize.x - 1));
        int y = Mathf.FloorToInt(Mathf.Min(GridSize.y * percentY, GridSize.y - 1));

        Debug.Log($"{name} NodeFromWorldPoint: ({x}, {y})");

        return WorldGrid[x, y];
    }

    public List<Node> GetNeighbours(Node node)
    {
        List<Node> neighbours = new List<Node>();

        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                if (x == 0 && y == 0)
                {
                    continue;
                }

                int px = node.GridX + x;
                int py = node.GridY + y;

                if (0 <= px && px < GridSize.x
                    && 0 <= py && py < GridSize.y)
                {
                    neighbours.Add(WorldGrid[px, py]);
                }
            }
        }

        return neighbours;
    }
}
