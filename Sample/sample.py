import asyncio
from mavsdk import System

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position OK")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.set_takeoff_altitude(5.0)  # meters
    await drone.action.takeoff()

    await asyncio.sleep(10)  # let it climb and hover

    # Print live telemetry once
    async for position in drone.telemetry.position():
        print(f"Altitude: {position.relative_altitude_m:.1f} m")
        break

    print("-- Landing")
    await drone.action.land()

    await asyncio.sleep(10)
    print("-- Done")

if __name__ == "__main__":
    asyncio.run(run())