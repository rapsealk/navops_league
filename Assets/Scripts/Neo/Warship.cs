using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Warship : MonoBehaviour
{
    public Color RendererColor;

    private Artillery[] artilleries;    // Weapon Systems Officer

    // Start is called before the first frame update
    void Start()
    {
        MeshRenderer[] meshRenderers = GetComponentsInChildren<MeshRenderer>();
        for (int i = 0; i < meshRenderers.Length; i++)
        {
            meshRenderers[i].material.color = RendererColor;
        }

        artilleries = GetComponentsInChildren<Artillery>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Mouse0))
        {
            for (int i = 0; i < artilleries.Length; i++)
            {
                artilleries[i].Fire();
            }
        }
    }

    /**
     * IArmedVehicle
     */
    /*
    public Transform GetTransform()
    {
        return transform;
    }
    */

    public void SetTargetPoint(Quaternion target)
    {
        for (int i = 0; i < artilleries.Length; i++)
        {
            artilleries[i].Rotate(target);
        }
    }
}
