using Pada1.BBCore;
using Pada1.BBCore.Tasks;
using UnityEngine;

namespace BBUnity.Actions
{
    /// <summary>
    /// It is an action to move the GameObject to a given position.
    /// </summary>
    [Action("Navigation/ReloadAmmo")]
    [Help("reload ammo int")]
    public class ReloadAmmo : GOAction
    {
        bool start;

        [InParam("ammo")]
        [Help("Target to check the distance")]
        public int ammo;

        /// <summary>Initialization Method of MoveToPosition.</summary>
        /// <remarks>Check if there is a NavMeshAgent to assign a default one and assign the destination to the NavMeshAgent the given position.</remarks>
        public override void OnStart()
        {
            Debug.Log("reloaded");
            ammo = 10;
        }

#if UNITY_5_6_OR_NEWER

#else
                navAgent.Resume();
#endif


        /// <summary>Method of Update of MoveToPosition </summary>
        /// <remarks>Check the status of the task, if it has traveled the road or is close to the goal it is completed
        /// and otherwise it will remain in operation.</remarks>
        public override TaskStatus OnUpdate()
        {

            //TaskStatus.
            return TaskStatus.COMPLETED;
        }

        /// <summary>Abort method of MoveToPosition.</summary>
        /// <remarks>When the task is aborted, it stops the navAgentMesh.</remarks>
        public override void OnAbort()
        {
#if UNITY_5_6_OR_NEWER

#else
            if (navAgent != null)
                navAgent.Stop();
#endif
        }
    }
}
