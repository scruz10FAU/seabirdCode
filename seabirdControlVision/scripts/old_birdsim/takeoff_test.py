import asyncio
from mavsdk import System
from mavsdk.action import ActionError

async def main():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected!")
            break

    print("Waiting for local position...")
    async for health in drone.telemetry.health():
        print(f"  gps_ok={health.is_global_position_ok}, local_ok={health.is_local_position_ok}")
        if health.is_local_position_ok:
            print("Local position OK!")
            break

    # Send force arm via raw MAVLink command (bypass preflight)
    await drone.manual_control.set_manual_control_input(0, 0, 0, 0)
    result = await drone.action.arm()
    print("Armed!")
    await asyncio.sleep(1)

    await drone.action.takeoff()
    print("Takeoff command sent — check Isaac viewport!")
    await asyncio.sleep(10)

    await drone.action.land()
    print("Landing!")
    await asyncio.sleep(5)

asyncio.run(main())
