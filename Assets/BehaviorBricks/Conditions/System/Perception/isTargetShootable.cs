using Pada1.BBCore;
using UnityEngine;
using System;

namespace BBUnity.Conditions
{
    /// <summary>
    /// It is a perception condition to check if the objective is close depending on a given distance.
    /// </summary>
    [Condition("Perception/IsTargetShootable")]
    [Help("Checks whether a target is close depending on a given distance")]
    public class IsTargetShootable : GOCondition
    {
        [InParam("Target")]
        public GameObject Target;

        [InParam("ShotSpeed")]
        public int ShotSpeed;

        [InParam("FireTransform")]
        public Transform FireTransform;

        public override bool Check()
        {

            float m_TargetDistance = Vector3.Distance(FireTransform.position, Target.transform.position);

            float calc = Physics.gravity.y * (m_TargetDistance * m_TargetDistance);//; +2 * 0
            double calc1 = Math.Sqrt((ShotSpeed * ShotSpeed * ShotSpeed * ShotSpeed) - Physics.gravity.y * (calc));
            double tangent = ((ShotSpeed * ShotSpeed) - calc1)
                / (Physics.gravity.y * m_TargetDistance);

            double Rad = Math.Atan(tangent);

            if (Math.Abs((float)Rad * Mathf.Rad2Deg) > 45 || float.IsNaN(Math.Abs((float)Rad * Mathf.Rad2Deg)))
            {
                //If not correct don't shoot
                Debug.Log("Can't fire, too far");
                return false;
            }



            return true;
        }
    }
}