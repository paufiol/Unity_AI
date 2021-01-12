using Pada1.BBCore;
using Pada1.BBCore.Tasks;
using UnityEngine;
using System;

namespace BBUnity.Actions
{
    /// <summary>
    /// It is an action to move the GameObject to a given position.
    /// </summary>
    [Action("Navigation/ShootBullet")]
    [Help("make an object wander")]
    public class ShootBullet : GOAction
    {
        bool start;

        [InParam("ammo")]
        [Help("Target to check the distance")]
        public int ammo;

        [InParam("Target")]
        public GameObject Target;

        [InParam("ShotSpeed")]
        public int ShotSpeed;

        [InParam("FireTransform")]
        public Transform FireTransform;

        [InParam("BulletPrefab")]
        public Rigidbody BulletPrefab;

        [InParam("TankTurret")]
        public GameObject TankTurret;

        [OutParam("ammoOut")]
        [Help("Target to check the distance")]
        public int ammoOut;

        public override void OnStart()
        {
            ammoOut = ammo;
           
        }
        public override TaskStatus OnUpdate()
        {
            ammoOut--;
            ammo--;

            Shoot();

            return TaskStatus.COMPLETED;
        }
        void Shoot()
        {
            Debug.Log("Shotbullet");

            float m_TargetDistance = Vector3.Distance(FireTransform.position, Target.transform.position);

            float calc = Physics.gravity.y * (m_TargetDistance * m_TargetDistance);//; +2 * 0
            double calc1 = Math.Sqrt((ShotSpeed * ShotSpeed * ShotSpeed * ShotSpeed) - Physics.gravity.y * (calc));
            double tangent = ((ShotSpeed * ShotSpeed) - calc1)
                / (Physics.gravity.y * m_TargetDistance);

            double Rad = Math.Atan(tangent);

            Debug.Log(-((float)Rad * Mathf.Rad2Deg));
            //Rotate cannon to desired angle
            TankTurret.transform.Rotate((float)Rad * Mathf.Rad2Deg, 0.0f, 0.0f);

            Rigidbody shellInstance =  UnityEngine.Object.Instantiate<Rigidbody>(BulletPrefab, FireTransform.position, FireTransform.rotation);

            //Shoot at set shot speed
            shellInstance.velocity = ShotSpeed * FireTransform.forward;
        }
    }
}