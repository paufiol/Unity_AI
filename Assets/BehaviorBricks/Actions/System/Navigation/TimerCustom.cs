using Pada1.BBCore;
using Pada1.BBCore.Tasks;
using UnityEngine;
using System;


namespace BBUnity.Actions
{
    /// <summary>
    /// It is an action to move the GameObject to a given position.
    /// </summary>
    [Action("Navigation/TimerCustom")]
    [Help("make an object wander")]
    public class TimerCustom : GOAction
    {
        bool start;

        [InParam("ShotDelay")]
        public float ShotDelay;

        [OutParam("ShotDelayReturn")]
        public float ShotDelayReturn;

        /// <summary>Initialization Method of MoveToPosition.</summary>
        /// <remarks>Check if there is a NavMeshAgent to assign a default one and assign the destination to the NavMeshAgent the given position.</remarks>
        public override void OnStart()
        {
            ShotDelayReturn = ShotDelay + Time.deltaTime;
        }


        /// <summary>Method of Update of MoveToPosition </summary>
        /// <remarks>Check the status of the task, if it has traveled the road or is close to the goal it is completed
        /// and otherwise it will remain in operation.</remarks>
        public override TaskStatus OnUpdate()
        {

            float i = UnityEngine.Random.Range(0.5f,1f);

            if (ShotDelay >= i)
            {
                ShotDelayReturn = 0;
                return TaskStatus.COMPLETED;
            }
            else
            {
                base.OnUpdate();
                return TaskStatus.FAILED;
            }
        }

        /// <summary>Abort method of MoveToPosition.</summary>
        /// <remarks>When the task is aborted, it stops the navAgentMesh.</remarks>
        public override void OnAbort()
        {

        }

    }
}