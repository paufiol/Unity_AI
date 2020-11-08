using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TextLookCamera : MonoBehaviour
{
    public Camera cameraLooked;
    // Start is called before the first frame update
    void Start()
    {
        cameraLooked = GameObject.Find("MainCamera").GetComponent<Camera>();
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 v = cameraLooked.transform.position - transform.position;
        v.x = v.z = 0.0f;
        transform.LookAt(cameraLooked.transform.position -v);
        transform.rotation = (cameraLooked.transform.rotation); // Take care about camera rotation
    }
}
