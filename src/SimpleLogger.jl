module SimpleLogger
using Revise
export set_log_level, debug, debug2, info, warn, error
export DEBUG, DEBUG2, INFO, WARN, ERROR
# Define log levels as integers
const DEBUG = 10
const DEBUG2 = 15
const INFO  = 20
const WARN  = 30
const ERROR = 40

# Global variable for the current log level.
# You can set it to one of the constants above.
global_log_level = INFO

"Set the global log level."
function set_log_level(level::Int)
    global global_log_level = level
end

"Internal function to print message if the given level is >= global level."
function log_message(level::Int, level_str::String, msg)
    if level >= global_log_level
        # Print with a timestamp if desired.
        println("[$(level_str)] ", msg)
    end
end

"Log a debug message."
debug(msg) = log_message(DEBUG, "DEBUG", msg)
"Log a debug2 message."
debug2(msg) = log_message(DEBUG2, "DEBUG2", msg)
"Log an info message."
info(msg)  = log_message(INFO,  "INFO", msg)
"Log a warning message."
warn(msg)  = log_message(WARN,  "WARN", msg)
"Log an error message."
error(msg) = log_message(ERROR, "ERROR", msg)

end # module