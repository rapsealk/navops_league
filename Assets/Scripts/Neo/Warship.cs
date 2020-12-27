using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Warship : MonoBehaviour
{
    public Color RendererColor;
    public ParticleSystem Explosion;

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

        ParticleSystem.MainModule explosionMainModule = Explosion.main;
        explosionMainModule.duration = 3f;

        /*
        int layerMask = 1 << 11;
        for (int i = 0; i < 8; i++)
        {
            Vector3 position = new Vector3(i * 10f, 0.1f, i * 10f);
            //Debug.Log($"LayerMask ({i * 10}, {i * 10}) {Geometry.ObjectExists(position, layerMask)}");
            //Debug.Log($"AllLayers ({i * 10}, {i * 10}) {Geometry.ObjectExists(position)}");

            RaycastHit hit;
            bool raycastResult = Physics.Raycast(Vector3.zero, position.normalized, out hit, i * 10, layerMask);
            Debug.Log($"Raycast ({i * 10}) {raycastResult} ({hit.distance})");
        }
        */
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

        //Vector3 rotation = new Vector3(-40f, -0.4f, 12f) - transform.position;
        //SetTargetPoint(Quaternion.Euler(rotation));
    }

    public void SetTargetPoint(Quaternion target)
    {
        for (int i = 0; i < artilleries.Length; i++)
        {
            artilleries[i].Rotate(target);
        }
    }

    public void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.name.StartsWith("Water"))
        {
            return;
        }

        Vector3 collisionVelocity = Vector3.zero;
        if (collision.rigidbody != null)
        {
            collisionVelocity = collision.rigidbody.velocity;
        }
        //
        Debug.Log($"Warship.OnCollisionEnter: {collision.collider} {collisionVelocity.magnitude} {collision.transform.position}");

        //Explosion.transform.position = collision.transform.position;
        //ParticleSystem explosion = Instantiate(ExplosionPrefab, collision.transform.position, collision.transform.rotation);
        //explosion.Stop();
        Explosion.transform.position = collision.transform.position;
        Explosion.transform.rotation = collision.transform.rotation;
        Explosion.Play();
    }

    public void OnTriggerEnter(Collider other)
    {
        //
    }
}
