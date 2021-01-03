using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovableTarget : MonoBehaviour
{
    public Warship Center;

    // Start is called before the first frame update
    void Start()
    {
        Vector3 position = Center.transform.position;
        position.z += 100f;
        transform.position = position;
        transform.rotation = Quaternion.Euler(Vector3.zero);
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = Center.transform.position + (transform.position - Center.transform.position).normalized * 30f;
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

        Vector3 rotation = Vector3.zero;
        rotation.y = Geometry.GetAngleBetween(Center.transform.position, transform.position);

        Center.SetTargetPoint(Quaternion.Euler(rotation));
    }
}
