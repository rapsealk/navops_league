using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GridField : MonoBehaviour
{
    public int Scale = 10;

    private Vector3Int Grid;

    // Start is called before the first frame update
    void Start()
    {
        Grid = Vector3Int.zero;
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public bool ObjectExists(Vector3Int grid, int layerMask = 1 << 11 /*LayerMask.NameToLayer("Terrain")*/)
    {
        RaycastHit hit;
        //Physics2D.Raycast(grid, Vector2.)
        return Physics.Raycast(grid, Vector3.forward, out hit, Mathf.Infinity, layerMask);
    }

    /*
    public static Vector3 operator+(GridField self, Vector3 other)
    {
        return Vector3.zero;
    }
    */

    public Vector3Int GetNextGrid(Vector2Int dir)
    {
        Grid += new Vector3Int(dir.x, 0, dir.y) * Scale;
        return Grid;
    }
}
