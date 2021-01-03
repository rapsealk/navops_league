using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovableTarget : MonoBehaviour
{
    public Warship Center;
    public float Distance = 100f;

    // Start is called before the first frame update
    void Start()
    {
        Vector3 position = Center.transform.position;
        position.z += Distance;
        transform.position = position;
        transform.rotation = Quaternion.Euler(Vector3.zero);
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = Center.transform.position + (transform.position - Center.transform.position).normalized * Distance;
        if (Input.GetKey(KeyCode.J))
        {
            transform.RotateAround(Center.transform.position, Center.transform.up, 100 * Time.deltaTime);
            transform.Rotate(Vector3.up, 25 * Time.deltaTime);
        }
        else if (Input.GetKey(KeyCode.K))
        {
            transform.RotateAround(Center.transform.position, Center.transform.up, -100 * Time.deltaTime);
            transform.Rotate(Vector3.up, -25 * Time.deltaTime);
        }
        else if (Input.GetKey(KeyCode.H))
        {
            Distance += 10f * Time.deltaTime;
        }
        else if (Input.GetKey(KeyCode.L))
        {
            Distance -= 10f * Time.deltaTime;
        }

        Vector3 rotation = Vector3.zero;
        rotation.y = Geometry.GetAngleBetween(Center.transform.position, transform.position);

        //
        Vector3 projectilexz = Center.transform.position;
        projectilexz.y = 0f;
        Vector3 targetxz = transform.position;
        targetxz.y = 0f;
        float r = Vector3.Distance(projectilexz, targetxz);
        float G = Physics.gravity.y;
        // float launchAngle = 0f;
        // float tanAlpha = Mathf.Tan(launchAngle * Math.Deg2Rad);
        // float launchAngle = Mathf.Atan(tanAlpha) * Mathf.Rad2Deg;
        //float h = 0f;
        //float vz = Mathf.Sqrt(G * r * r / (2f * (h - r * tanAlpha)));
        //Mathf.Pow(vz, 2f) = G * r * r / (2f * r * tanAlpha);
        //tanAlpha = (G * r) / (Mathf.Pow(vz, 2f) * 2f);
        //angle = Mathf.Atan(tanAlpha * Mathf.Rad2Deg);
        //angle = Mathf.Atan((G * r) / (Mathf.Pow(vz, 2f) * 2f) * Mathf.Rad2Deg);
        //float vy = tanAlpha * vz;
        float vz = 8000f;
        //float vy = 100f;
        //Vector3 v = Center.transform.TransformDirection(new Vector3(0f, vy, vz));

        // rotation.x = Mathf.Atan((G * r) / (vz * Mathf.Pow(vz, 2f) * 2f)) * Mathf.Rad2Deg;
        rotation.x = Mathf.Atan((G * r) / (vz * 2f)) * Mathf.Rad2Deg;   // max: 140

        Center.SetTargetPoint(Quaternion.Euler(rotation));
    }
}
