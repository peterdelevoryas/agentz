const std = @import("std");
const openai = @import("agentz").openai;
const tui = @import("tui.zig");

pub fn main(init: std.process.Init) !void {
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args.deinit();

    _ = args.skip();
    const first = args.next();
    if (first) |arg| {
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            return printUsage(init.io);
        }
        if (std.mem.eql(u8, arg, "once")) {
            const prompt = try joinArgs(init.gpa, &args, args.next());
            defer if (prompt) |text| init.gpa.free(text);
            return runOnce(init, prompt orelse "Reply with exactly: API wiring works.");
        }
        if (std.mem.eql(u8, arg, "tui")) {
            const initial_prompt = try joinArgs(init.gpa, &args, null);
            defer if (initial_prompt) |prompt| init.gpa.free(prompt);
            return tui.run(init, .{ .initial_prompt = initial_prompt });
        }

        const initial_prompt = try joinArgs(init.gpa, &args, arg);
        defer if (initial_prompt) |prompt| init.gpa.free(prompt);
        return tui.run(init, .{ .initial_prompt = initial_prompt });
    }

    return tui.run(init, .{});
}

fn runOnce(init: std.process.Init, prompt: []const u8) !void {
    var client = try openai.Client.fromEnv(init.gpa, init.io, init.environ_map);
    defer client.deinit();

    var response = try client.createResponse(
        init.gpa,
        openai.ResponsesRequest.init(openai.DEFAULT_MODEL, .{ .text = prompt }),
    );
    defer response.deinit();

    const output_text = try response.outputTextAlloc(init.gpa);
    defer init.gpa.free(output_text);

    try std.Io.File.writeStreamingAll(.stdout(), init.io, output_text);
    try std.Io.File.writeStreamingAll(.stdout(), init.io, "\n");
}

fn joinArgs(
    gpa: std.mem.Allocator,
    args: *std.process.Args.Iterator,
    first: ?[]const u8,
) !?[]u8 {
    var joined: std.ArrayList(u8) = .empty;
    errdefer joined.deinit(gpa);

    if (first) |value| try joined.appendSlice(gpa, value);
    while (args.next()) |arg| {
        if (joined.items.len != 0) try joined.append(gpa, ' ');
        try joined.appendSlice(gpa, arg);
    }

    if (joined.items.len == 0) return null;
    return try joined.toOwnedSlice(gpa);
}

fn printUsage(io: std.Io) !void {
    try std.Io.File.writeStreamingAll(.stdout(), io,
        \\agentz
        \\  Launch the local TUI prototype.
        \\
        \\agentz [initial prompt words...]
        \\  Launch the TUI with the composer pre-filled.
        \\
        \\agentz once [prompt]
        \\  Run the old one-shot OpenAI Responses request.
        \\
    );
}
