//! DO NOT EDIT
//! Auto-generated code with script/codegen.py
//! For regeneration, run following bash script in <path-to-bake> :`uv run script/codegen.py`

#[allow(unused)]
use crate::env::gym::env::{GymnasiumEnvironmentConfig, BatchedGymnasiumEnvironmentConfig};
#[allow(unused)]
use crate::core::env::space::{DiscreteSpace, ContinuousSpace, ByteSpace};
#[allow(unused)]
use crate::core::env::space::{BatchedDiscreteSpace, BatchedContinuousSpace, BatchedByteSpace};

impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Adventure-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn adventure_preprocessed() -> Self {
        Self::new(ADVENTURE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/AirRaid-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn air_raid_preprocessed() -> Self {
        Self::new(AIR_RAID_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Alien-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn alien_preprocessed() -> Self {
        Self::new(ALIEN_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Amidar-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn amidar_preprocessed() -> Self {
        Self::new(AMIDAR_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Assault-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn assault_preprocessed() -> Self {
        Self::new(ASSAULT_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Asterix-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn asterix_preprocessed() -> Self {
        Self::new(ASTERIX_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Asteroids-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn asteroids_preprocessed() -> Self {
        Self::new(ASTEROIDS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Atlantis2-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn atlantis2_preprocessed() -> Self {
        Self::new(ATLANTIS2_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Atlantis-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn atlantis_preprocessed() -> Self {
        Self::new(ATLANTIS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/BankHeist-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn bank_heist_preprocessed() -> Self {
        Self::new(BANK_HEIST_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/BasicMath-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn basic_math_preprocessed() -> Self {
        Self::new(BASIC_MATH_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/BattleZone-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn battle_zone_preprocessed() -> Self {
        Self::new(BATTLE_ZONE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/BeamRider-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn beam_rider_preprocessed() -> Self {
        Self::new(BEAM_RIDER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Berzerk-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn berzerk_preprocessed() -> Self {
        Self::new(BERZERK_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Blackjack-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn blackjack_preprocessed() -> Self {
        Self::new(BLACKJACK_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Bowling-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn bowling_preprocessed() -> Self {
        Self::new(BOWLING_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Boxing-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn boxing_preprocessed() -> Self {
        Self::new(BOXING_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Breakout-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn breakout_preprocessed() -> Self {
        Self::new(BREAKOUT_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Carnival-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn carnival_preprocessed() -> Self {
        Self::new(CARNIVAL_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Casino-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn casino_preprocessed() -> Self {
        Self::new(CASINO_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Centipede-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn centipede_preprocessed() -> Self {
        Self::new(CENTIPEDE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/ChopperCommand-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn chopper_command_preprocessed() -> Self {
        Self::new(CHOPPER_COMMAND_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/CrazyClimber-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn crazy_climber_preprocessed() -> Self {
        Self::new(CRAZY_CLIMBER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Crossbow-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn crossbow_preprocessed() -> Self {
        Self::new(CROSSBOW_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Darkchambers-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn darkchambers_preprocessed() -> Self {
        Self::new(DARKCHAMBERS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Defender-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn defender_preprocessed() -> Self {
        Self::new(DEFENDER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/DemonAttack-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn demon_attack_preprocessed() -> Self {
        Self::new(DEMON_ATTACK_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/DonkeyKong-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn donkey_kong_preprocessed() -> Self {
        Self::new(DONKEY_KONG_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/DoubleDunk-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn double_dunk_preprocessed() -> Self {
        Self::new(DOUBLE_DUNK_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Earthworld-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn earthworld_preprocessed() -> Self {
        Self::new(EARTHWORLD_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/ElevatorAction-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn elevator_action_preprocessed() -> Self {
        Self::new(ELEVATOR_ACTION_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Enduro-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn enduro_preprocessed() -> Self {
        Self::new(ENDURO_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Entombed-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn entombed_preprocessed() -> Self {
        Self::new(ENTOMBED_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Et-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn et_preprocessed() -> Self {
        Self::new(ET_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/FishingDerby-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn fishing_derby_preprocessed() -> Self {
        Self::new(FISHING_DERBY_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/FlagCapture-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn flag_capture_preprocessed() -> Self {
        Self::new(FLAG_CAPTURE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Freeway-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn freeway_preprocessed() -> Self {
        Self::new(FREEWAY_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Frogger-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn frogger_preprocessed() -> Self {
        Self::new(FROGGER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Frostbite-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn frostbite_preprocessed() -> Self {
        Self::new(FROSTBITE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Galaxian-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn galaxian_preprocessed() -> Self {
        Self::new(GALAXIAN_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Gopher-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn gopher_preprocessed() -> Self {
        Self::new(GOPHER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Gravitar-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn gravitar_preprocessed() -> Self {
        Self::new(GRAVITAR_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Hangman-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn hangman_preprocessed() -> Self {
        Self::new(HANGMAN_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/HauntedHouse-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn haunted_house_preprocessed() -> Self {
        Self::new(HAUNTED_HOUSE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Hero-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn hero_preprocessed() -> Self {
        Self::new(HERO_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/HumanCannonball-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn human_cannonball_preprocessed() -> Self {
        Self::new(HUMAN_CANNONBALL_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/IceHockey-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn ice_hockey_preprocessed() -> Self {
        Self::new(ICE_HOCKEY_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Jamesbond-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn jamesbond_preprocessed() -> Self {
        Self::new(JAMESBOND_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/JourneyEscape-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn journey_escape_preprocessed() -> Self {
        Self::new(JOURNEY_ESCAPE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Kaboom-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn kaboom_preprocessed() -> Self {
        Self::new(KABOOM_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Kangaroo-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn kangaroo_preprocessed() -> Self {
        Self::new(KANGAROO_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/KeystoneKapers-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn keystone_kapers_preprocessed() -> Self {
        Self::new(KEYSTONE_KAPERS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/KingKong-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn king_kong_preprocessed() -> Self {
        Self::new(KING_KONG_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Klax-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn klax_preprocessed() -> Self {
        Self::new(KLAX_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Koolaid-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn koolaid_preprocessed() -> Self {
        Self::new(KOOLAID_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Krull-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn krull_preprocessed() -> Self {
        Self::new(KRULL_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/KungFuMaster-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn kung_fu_master_preprocessed() -> Self {
        Self::new(KUNG_FU_MASTER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/LaserGates-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn laser_gates_preprocessed() -> Self {
        Self::new(LASER_GATES_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/LostLuggage-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn lost_luggage_preprocessed() -> Self {
        Self::new(LOST_LUGGAGE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/MarioBros-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn mario_bros_preprocessed() -> Self {
        Self::new(MARIO_BROS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/MiniatureGolf-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn miniature_golf_preprocessed() -> Self {
        Self::new(MINIATURE_GOLF_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/MontezumaRevenge-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn montezuma_revenge_preprocessed() -> Self {
        Self::new(MONTEZUMA_REVENGE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/MrDo-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn mr_do_preprocessed() -> Self {
        Self::new(MR_DO_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/MsPacman-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn ms_pacman_preprocessed() -> Self {
        Self::new(MS_PACMAN_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/NameThisGame-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn name_this_game_preprocessed() -> Self {
        Self::new(NAME_THIS_GAME_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Othello-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn othello_preprocessed() -> Self {
        Self::new(OTHELLO_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Pacman-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pacman_preprocessed() -> Self {
        Self::new(PACMAN_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Phoenix-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn phoenix_preprocessed() -> Self {
        Self::new(PHOENIX_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Pitfall2-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pitfall2_preprocessed() -> Self {
        Self::new(PITFALL2_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Pitfall-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pitfall_preprocessed() -> Self {
        Self::new(PITFALL_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Pong-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pong_preprocessed() -> Self {
        Self::new(PONG_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Pooyan-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn pooyan_preprocessed() -> Self {
        Self::new(POOYAN_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/PrivateEye-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn private_eye_preprocessed() -> Self {
        Self::new(PRIVATE_EYE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Qbert-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn qbert_preprocessed() -> Self {
        Self::new(QBERT_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Riverraid-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn riverraid_preprocessed() -> Self {
        Self::new(RIVERRAID_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/RoadRunner-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn road_runner_preprocessed() -> Self {
        Self::new(ROAD_RUNNER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Robotank-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn robotank_preprocessed() -> Self {
        Self::new(ROBOTANK_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Seaquest-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn seaquest_preprocessed() -> Self {
        Self::new(SEAQUEST_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/SirLancelot-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn sir_lancelot_preprocessed() -> Self {
        Self::new(SIR_LANCELOT_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Skiing-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn skiing_preprocessed() -> Self {
        Self::new(SKIING_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Solaris-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn solaris_preprocessed() -> Self {
        Self::new(SOLARIS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/SpaceInvaders-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn space_invaders_preprocessed() -> Self {
        Self::new(SPACE_INVADERS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/SpaceWar-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn space_war_preprocessed() -> Self {
        Self::new(SPACE_WAR_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/StarGunner-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn star_gunner_preprocessed() -> Self {
        Self::new(STAR_GUNNER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Superman-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn superman_preprocessed() -> Self {
        Self::new(SUPERMAN_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Surround-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn surround_preprocessed() -> Self {
        Self::new(SURROUND_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Tennis-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn tennis_preprocessed() -> Self {
        Self::new(TENNIS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Tetris-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn tetris_preprocessed() -> Self {
        Self::new(TETRIS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/TicTacToe3D-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn tic_tac_toe3d_preprocessed() -> Self {
        Self::new(TIC_TAC_TOE3D_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/TimePilot-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn time_pilot_preprocessed() -> Self {
        Self::new(TIME_PILOT_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Trondead-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn trondead_preprocessed() -> Self {
        Self::new(TRONDEAD_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Turmoil-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn turmoil_preprocessed() -> Self {
        Self::new(TURMOIL_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Tutankham-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn tutankham_preprocessed() -> Self {
        Self::new(TUTANKHAM_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/UpNDown-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn up_n_down_preprocessed() -> Self {
        Self::new(UP_N_DOWN_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Venture-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn venture_preprocessed() -> Self {
        Self::new(VENTURE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/VideoChess-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn video_chess_preprocessed() -> Self {
        Self::new(VIDEO_CHESS_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/VideoCube-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn video_cube_preprocessed() -> Self {
        Self::new(VIDEO_CUBE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/VideoPinball-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn video_pinball_preprocessed() -> Self {
        Self::new(VIDEO_PINBALL_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/WizardOfWor-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn wizard_of_wor_preprocessed() -> Self {
        Self::new(WIZARD_OF_WOR_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/WordZapper-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn word_zapper_preprocessed() -> Self {
        Self::new(WORD_ZAPPER_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/YarsRevenge-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn yars_revenge_preprocessed() -> Self {
        Self::new(YARS_REVENGE_SPEC)
    }
}
impl GymnasiumEnvironmentConfig<ByteSpace<3>, DiscreteSpace> {
    /// `ALE/Zaxxon-v5` environment, preprocessed with `frame_skip = 4` and `stack_size = 4`
    /// - `ObsSpace`: `ByteSpace<3>`
    /// - `ActionSpace`: `DiscreteSpace`
    pub fn zaxxon_preprocessed() -> Self {
        Self::new(ZAXXON_SPEC)
    }
}

const ADVENTURE_SPEC: &'static str = r#"{"id": "ALE/Adventure-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "adventure", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const AIR_RAID_SPEC: &'static str = r#"{"id": "ALE/AirRaid-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "air_raid", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ALIEN_SPEC: &'static str = r#"{"id": "ALE/Alien-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "alien", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const AMIDAR_SPEC: &'static str = r#"{"id": "ALE/Amidar-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "amidar", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ASSAULT_SPEC: &'static str = r#"{"id": "ALE/Assault-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "assault", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ASTERIX_SPEC: &'static str = r#"{"id": "ALE/Asterix-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "asterix", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ASTEROIDS_SPEC: &'static str = r#"{"id": "ALE/Asteroids-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "asteroids", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ATLANTIS2_SPEC: &'static str = r#"{"id": "ALE/Atlantis2-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "atlantis2", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ATLANTIS_SPEC: &'static str = r#"{"id": "ALE/Atlantis-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "atlantis", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const BANK_HEIST_SPEC: &'static str = r#"{"id": "ALE/BankHeist-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "bank_heist", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const BASIC_MATH_SPEC: &'static str = r#"{"id": "ALE/BasicMath-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "basic_math", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const BATTLE_ZONE_SPEC: &'static str = r#"{"id": "ALE/BattleZone-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "battle_zone", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const BEAM_RIDER_SPEC: &'static str = r#"{"id": "ALE/BeamRider-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "beam_rider", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const BERZERK_SPEC: &'static str = r#"{"id": "ALE/Berzerk-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "berzerk", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const BLACKJACK_SPEC: &'static str = r#"{"id": "ALE/Blackjack-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "blackjack", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const BOWLING_SPEC: &'static str = r#"{"id": "ALE/Bowling-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "bowling", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const BOXING_SPEC: &'static str = r#"{"id": "ALE/Boxing-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "boxing", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const BREAKOUT_SPEC: &'static str = r#"{"id": "ALE/Breakout-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "breakout", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const CARNIVAL_SPEC: &'static str = r#"{"id": "ALE/Carnival-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "carnival", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const CASINO_SPEC: &'static str = r#"{"id": "ALE/Casino-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "casino", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const CENTIPEDE_SPEC: &'static str = r#"{"id": "ALE/Centipede-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "centipede", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const CHOPPER_COMMAND_SPEC: &'static str = r#"{"id": "ALE/ChopperCommand-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "chopper_command", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const CRAZY_CLIMBER_SPEC: &'static str = r#"{"id": "ALE/CrazyClimber-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "crazy_climber", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const CROSSBOW_SPEC: &'static str = r#"{"id": "ALE/Crossbow-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "crossbow", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const DARKCHAMBERS_SPEC: &'static str = r#"{"id": "ALE/Darkchambers-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "darkchambers", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const DEFENDER_SPEC: &'static str = r#"{"id": "ALE/Defender-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "defender", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const DEMON_ATTACK_SPEC: &'static str = r#"{"id": "ALE/DemonAttack-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "demon_attack", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const DONKEY_KONG_SPEC: &'static str = r#"{"id": "ALE/DonkeyKong-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "donkey_kong", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const DOUBLE_DUNK_SPEC: &'static str = r#"{"id": "ALE/DoubleDunk-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "double_dunk", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const EARTHWORLD_SPEC: &'static str = r#"{"id": "ALE/Earthworld-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "earthworld", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ELEVATOR_ACTION_SPEC: &'static str = r#"{"id": "ALE/ElevatorAction-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "elevator_action", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ENDURO_SPEC: &'static str = r#"{"id": "ALE/Enduro-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "enduro", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ENTOMBED_SPEC: &'static str = r#"{"id": "ALE/Entombed-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "entombed", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ET_SPEC: &'static str = r#"{"id": "ALE/Et-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "et", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const FISHING_DERBY_SPEC: &'static str = r#"{"id": "ALE/FishingDerby-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "fishing_derby", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const FLAG_CAPTURE_SPEC: &'static str = r#"{"id": "ALE/FlagCapture-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "flag_capture", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const FREEWAY_SPEC: &'static str = r#"{"id": "ALE/Freeway-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "freeway", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const FROGGER_SPEC: &'static str = r#"{"id": "ALE/Frogger-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "frogger", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const FROSTBITE_SPEC: &'static str = r#"{"id": "ALE/Frostbite-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "frostbite", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const GALAXIAN_SPEC: &'static str = r#"{"id": "ALE/Galaxian-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "galaxian", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const GOPHER_SPEC: &'static str = r#"{"id": "ALE/Gopher-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "gopher", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const GRAVITAR_SPEC: &'static str = r#"{"id": "ALE/Gravitar-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "gravitar", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const HANGMAN_SPEC: &'static str = r#"{"id": "ALE/Hangman-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "hangman", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const HAUNTED_HOUSE_SPEC: &'static str = r#"{"id": "ALE/HauntedHouse-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "haunted_house", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const HERO_SPEC: &'static str = r#"{"id": "ALE/Hero-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "hero", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const HUMAN_CANNONBALL_SPEC: &'static str = r#"{"id": "ALE/HumanCannonball-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "human_cannonball", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ICE_HOCKEY_SPEC: &'static str = r#"{"id": "ALE/IceHockey-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "ice_hockey", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const JAMESBOND_SPEC: &'static str = r#"{"id": "ALE/Jamesbond-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "jamesbond", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const JOURNEY_ESCAPE_SPEC: &'static str = r#"{"id": "ALE/JourneyEscape-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "journey_escape", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const KABOOM_SPEC: &'static str = r#"{"id": "ALE/Kaboom-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "kaboom", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const KANGAROO_SPEC: &'static str = r#"{"id": "ALE/Kangaroo-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "kangaroo", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const KEYSTONE_KAPERS_SPEC: &'static str = r#"{"id": "ALE/KeystoneKapers-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "keystone_kapers", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const KING_KONG_SPEC: &'static str = r#"{"id": "ALE/KingKong-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "king_kong", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const KLAX_SPEC: &'static str = r#"{"id": "ALE/Klax-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "klax", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const KOOLAID_SPEC: &'static str = r#"{"id": "ALE/Koolaid-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "koolaid", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const KRULL_SPEC: &'static str = r#"{"id": "ALE/Krull-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "krull", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const KUNG_FU_MASTER_SPEC: &'static str = r#"{"id": "ALE/KungFuMaster-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "kung_fu_master", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const LASER_GATES_SPEC: &'static str = r#"{"id": "ALE/LaserGates-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "laser_gates", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const LOST_LUGGAGE_SPEC: &'static str = r#"{"id": "ALE/LostLuggage-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "lost_luggage", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const MARIO_BROS_SPEC: &'static str = r#"{"id": "ALE/MarioBros-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "mario_bros", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const MINIATURE_GOLF_SPEC: &'static str = r#"{"id": "ALE/MiniatureGolf-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "miniature_golf", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const MONTEZUMA_REVENGE_SPEC: &'static str = r#"{"id": "ALE/MontezumaRevenge-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "montezuma_revenge", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const MR_DO_SPEC: &'static str = r#"{"id": "ALE/MrDo-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "mr_do", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const MS_PACMAN_SPEC: &'static str = r#"{"id": "ALE/MsPacman-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "ms_pacman", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const NAME_THIS_GAME_SPEC: &'static str = r#"{"id": "ALE/NameThisGame-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "name_this_game", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const OTHELLO_SPEC: &'static str = r#"{"id": "ALE/Othello-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "othello", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const PACMAN_SPEC: &'static str = r#"{"id": "ALE/Pacman-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pacman", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const PHOENIX_SPEC: &'static str = r#"{"id": "ALE/Phoenix-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "phoenix", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const PITFALL2_SPEC: &'static str = r#"{"id": "ALE/Pitfall2-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pitfall2", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const PITFALL_SPEC: &'static str = r#"{"id": "ALE/Pitfall-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pitfall", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const PONG_SPEC: &'static str = r#"{"id": "ALE/Pong-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pong", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const POOYAN_SPEC: &'static str = r#"{"id": "ALE/Pooyan-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "pooyan", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const PRIVATE_EYE_SPEC: &'static str = r#"{"id": "ALE/PrivateEye-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "private_eye", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const QBERT_SPEC: &'static str = r#"{"id": "ALE/Qbert-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "qbert", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const RIVERRAID_SPEC: &'static str = r#"{"id": "ALE/Riverraid-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "riverraid", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ROAD_RUNNER_SPEC: &'static str = r#"{"id": "ALE/RoadRunner-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "road_runner", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ROBOTANK_SPEC: &'static str = r#"{"id": "ALE/Robotank-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "robotank", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const SEAQUEST_SPEC: &'static str = r#"{"id": "ALE/Seaquest-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "seaquest", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const SIR_LANCELOT_SPEC: &'static str = r#"{"id": "ALE/SirLancelot-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "sir_lancelot", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const SKIING_SPEC: &'static str = r#"{"id": "ALE/Skiing-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "skiing", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const SOLARIS_SPEC: &'static str = r#"{"id": "ALE/Solaris-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "solaris", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const SPACE_INVADERS_SPEC: &'static str = r#"{"id": "ALE/SpaceInvaders-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "space_invaders", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const SPACE_WAR_SPEC: &'static str = r#"{"id": "ALE/SpaceWar-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "space_war", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const STAR_GUNNER_SPEC: &'static str = r#"{"id": "ALE/StarGunner-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "star_gunner", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const SUPERMAN_SPEC: &'static str = r#"{"id": "ALE/Superman-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "superman", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const SURROUND_SPEC: &'static str = r#"{"id": "ALE/Surround-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "surround", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const TENNIS_SPEC: &'static str = r#"{"id": "ALE/Tennis-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "tennis", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const TETRIS_SPEC: &'static str = r#"{"id": "ALE/Tetris-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "tetris", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const TIC_TAC_TOE3D_SPEC: &'static str = r#"{"id": "ALE/TicTacToe3D-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "tic_tac_toe_3d", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const TIME_PILOT_SPEC: &'static str = r#"{"id": "ALE/TimePilot-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "time_pilot", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const TRONDEAD_SPEC: &'static str = r#"{"id": "ALE/Trondead-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "trondead", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const TURMOIL_SPEC: &'static str = r#"{"id": "ALE/Turmoil-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "turmoil", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const TUTANKHAM_SPEC: &'static str = r#"{"id": "ALE/Tutankham-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "tutankham", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const UP_N_DOWN_SPEC: &'static str = r#"{"id": "ALE/UpNDown-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "up_n_down", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const VENTURE_SPEC: &'static str = r#"{"id": "ALE/Venture-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "venture", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const VIDEO_CHESS_SPEC: &'static str = r#"{"id": "ALE/VideoChess-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "video_chess", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const VIDEO_CUBE_SPEC: &'static str = r#"{"id": "ALE/VideoCube-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "video_cube", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const VIDEO_PINBALL_SPEC: &'static str = r#"{"id": "ALE/VideoPinball-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "video_pinball", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const WIZARD_OF_WOR_SPEC: &'static str = r#"{"id": "ALE/WizardOfWor-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "wizard_of_wor", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const WORD_ZAPPER_SPEC: &'static str = r#"{"id": "ALE/WordZapper-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "word_zapper", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const YARS_REVENGE_SPEC: &'static str = r#"{"id": "ALE/YarsRevenge-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "yars_revenge", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;


const ZAXXON_SPEC: &'static str = r#"{"id": "ALE/Zaxxon-v5", "entry_point": "ale_py.env:AtariEnv", "reward_threshold": null, "nondeterministic": false, "max_episode_steps": null, "order_enforce": true, "disable_env_checker": false, "kwargs": {"game": "zaxxon", "repeat_action_probability": 0.25, "full_action_space": false, "frameskip": 1, "max_num_frames_per_episode": 108000}, "additional_wrappers": [{"name": "AtariPreprocessing", "entry_point": "gymnasium.wrappers.atari_preprocessing:AtariPreprocessing", "kwargs": {"noop_max": 30, "frame_skip": 4, "screen_size": 84, "terminal_on_life_loss": false, "grayscale_obs": true, "grayscale_newaxis": false, "scale_obs": false}}, {"name": "FrameStackObservation", "entry_point": "gymnasium.wrappers.stateful_observation:FrameStackObservation", "kwargs": {"stack_size": 4, "padding_type": "reset"}}], "vector_entry_point": "ale_py.vector_env:AtariVectorEnv"}"#;

