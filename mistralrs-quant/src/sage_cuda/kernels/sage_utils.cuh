/*
 * Copyright (c) 2024 by SageAttention team. Vendored into Arc with torch
 * dependency removed; all runtime checks moved to Rust caller.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#pragma once

#include <cstdint>
#include <cstdlib>

#ifndef div_ceil
#define div_ceil(M, N) (((M) + (N)-1) / (N))
#endif
