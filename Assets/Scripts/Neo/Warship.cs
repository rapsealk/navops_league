using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Warship : MonoBehaviour
{
    public Color RendererColor;
    public ParticleSystem Explosion;
    public GameObject TorpedoPrefab;
    public Transform TargetObject;

    private Quaternion cameraQuaternion;
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
        explosionMainModule.loop = false;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Mouse0))
        {
            FireMainBattery();
        }
        else if (Input.GetKeyDown(KeyCode.Mouse1))
        {
            Debug.Log("KeyCode.Mouse1");

            // if (Application.platform == RuntimePlatform.WindowsEditor) { }

            // TODO: Animation
            /* Human Controller
            Vector3 pointer = Input.mousePosition;
            Ray cast = Camera.main.ScreenPointToRay(pointer);
            int waterLayerMask = 1 << 4;
            if (Physics.Raycast(cast, out RaycastHit hit, Mathf.Infinity, waterLayerMask))
            {
                Debug.Log($"RaycastHit: {hit.point}");
                FireTorpedoAt(hit.point, cameraQuaternion.eulerAngles);
            }
            */
            // API Controller
            float y = Geometry.GetAngleBetween(transform.position, TargetObject.transform.position);
            Vector3 rotation = new Vector3(0f, y, 0f);
            FireTorpedoAt(TargetObject.transform.position, rotation);
        }
        else if (Input.GetKeyDown(KeyCode.Mouse2))  // Wheel
        {
            Debug.Log($"Mouse..");
        }
    }

    public void SetTargetPoint(Quaternion target)
    {
        cameraQuaternion = target;

        for (int i = 0; i < artilleries.Length; i++)
        {
            artilleries[i].Rotate(target);
        }
    }

    public void FireMainBattery()
    {
        for (int i = 0; i < artilleries.Length; i++)
        {
            artilleries[i].Fire();
        }
    }

    public void FireTorpedoAt(Vector3 position, Vector3 rotation)
    {
        Vector3 releasePoint = transform.position + (position - transform.position).normalized * 8f;
        releasePoint.y = 0f;

        rotation.x = 90f;
        rotation.z = 0f;

        GameObject _ = Instantiate(TorpedoPrefab, releasePoint, Quaternion.Euler(rotation));
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

        Debug.Log($"Warship.OnCollisionEnter: {collision.collider} {collisionVelocity.magnitude} {collision.transform.position}");

        Explosion.transform.position = collision.transform.position;
        Explosion.transform.rotation = collision.transform.rotation;
        Explosion.Play();

        if (collision.collider.tag.Equals("Torpedo"))
        {
            // Destroy;
        }
    }

    public void OnTriggerEnter(Collider other)
    {
        Debug.Log($"Warship.OnTriggerEnter(other: {other})");

        Explosion.transform.position = other.transform.position;
        Explosion.transform.rotation = other.transform.rotation;
        Explosion.Play();
    }
}
