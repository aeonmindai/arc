//! PHP trajectories (4).

use super::{assemble, assistant_call, assistant_text, src, tool, user};
use crate::trajectory::Trajectory;
use serde_json::json;

pub fn all() -> Vec<Trajectory> {
    vec![
        php_034_laravel_route(),
        php_035_eloquent_scope(),
        php_036_symfony_command(),
        php_037_phpunit_data_provider(),
    ]
}

fn php_034_laravel_route() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a GET /healthz route in routes/api.php that returns ['status' => 'ok'] with response()->json. Use Route::get.",
    ));
    turns.push(assistant_call(
        "Reading routes/api.php.",
        "call_001",
        "read_file",
        json!({"path":"routes/api.php"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "<?php\nuse Illuminate\\Support\\Facades\\Route;\n\nRoute::middleware('auth:sanctum')->get('/user', function (Request $request) {\n    return $request->user();\n});\n",
    ));
    turns.push(assistant_call(
        "Adding the route.",
        "call_002",
        "write_file",
        json!({
            "path":"routes/api.php",
            "contents":"<?php\nuse Illuminate\\Support\\Facades\\Route;\n\nRoute::middleware('auth:sanctum')->get('/user', function (Request $request) {\n    return $request->user();\n});\n\nRoute::get('/healthz', fn () => response()->json(['status' => 'ok']));\n"
        }),
        &turns,
    ));
    turns.push(tool("call_002", "wrote 296 bytes to routes/api.php"));
    turns.push(assistant_text("Healthz route registered.", &turns));

    assemble(
        "agentperf_tuning_034",
        "php",
        src("github.com/laravel/laravel", "MIT", "11.x"),
        turns,
    )
}

fn php_035_eloquent_scope() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Add a local scope `published` to the Article Eloquent model that filters `published_at` <= now() AND `status` = 'published'.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"app/Models/Article.php"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "<?php\nnamespace App\\Models;\nuse Illuminate\\Database\\Eloquent\\Model;\n\nclass Article extends Model {\n    protected $fillable = ['title', 'body', 'status', 'published_at'];\n}\n",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_002",
        "write_file",
        json!({
            "path":"app/Models/Article.php",
            "contents":"<?php\nnamespace App\\Models;\nuse Illuminate\\Database\\Eloquent\\Builder;\nuse Illuminate\\Database\\Eloquent\\Model;\n\nclass Article extends Model {\n    protected $fillable = ['title', 'body', 'status', 'published_at'];\n\n    public function scopePublished(Builder $query): Builder {\n        return $query->where('status', 'published')\n                     ->where('published_at', '<=', now());\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 410 bytes to app/Models/Article.php",
    ));
    turns.push(assistant_text(
        "Use as `Article::published()->get()`.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_035",
        "php",
        src("github.com/laravel/framework", "MIT", "11.x"),
        turns,
    )
}

fn php_036_symfony_command() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Create a Symfony console command `app:purge-expired` that deletes Token rows where expires_at < now(). Use EntityManagerInterface DI.",
    ));
    turns.push(assistant_call(
        "Writing.",
        "call_001",
        "write_file",
        json!({
            "path":"src/Command/PurgeExpiredCommand.php",
            "contents":"<?php\nnamespace App\\Command;\n\nuse App\\Entity\\Token;\nuse Doctrine\\ORM\\EntityManagerInterface;\nuse Symfony\\Component\\Console\\Attribute\\AsCommand;\nuse Symfony\\Component\\Console\\Command\\Command;\nuse Symfony\\Component\\Console\\Input\\InputInterface;\nuse Symfony\\Component\\Console\\Output\\OutputInterface;\n\n#[AsCommand(name: 'app:purge-expired', description: 'Delete expired tokens')]\nfinal class PurgeExpiredCommand extends Command {\n    public function __construct(private EntityManagerInterface $em) { parent::__construct(); }\n\n    protected function execute(InputInterface $input, OutputInterface $output): int {\n        $deleted = $this->em->createQuery(\n            'DELETE FROM ' . Token::class . ' t WHERE t.expiresAt < :now'\n        )->setParameter('now', new \\DateTimeImmutable())->execute();\n        $output->writeln(sprintf('Deleted %d tokens', $deleted));\n        return Command::SUCCESS;\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "wrote 1030 bytes to src/Command/PurgeExpiredCommand.php",
    ));
    turns.push(assistant_text(
        "Command registered via `#[AsCommand]`. Run with `bin/console app:purge-expired`.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_036",
        "php",
        src("github.com/symfony/symfony", "MIT", "7.x"),
        turns,
    )
}

fn php_037_phpunit_data_provider() -> Trajectory {
    let mut turns = Vec::new();
    turns.push(user(
        "Refactor StringReverseTest.php from three separate test methods into a single test method backed by a data provider.",
    ));
    turns.push(assistant_call(
        "Reading.",
        "call_001",
        "read_file",
        json!({"path":"tests/StringReverseTest.php"}),
        &turns,
    ));
    turns.push(tool(
        "call_001",
        "<?php\nuse PHPUnit\\Framework\\TestCase;\n\nclass StringReverseTest extends TestCase {\n    public function testEmpty()  { $this->assertSame('', str_reverse_unicode('')); }\n    public function testAscii()  { $this->assertSame('olleh', str_reverse_unicode('hello')); }\n    public function testAccent() { $this->assertSame('olléh', str_reverse_unicode('héllo')); }\n}\n",
    ));
    turns.push(assistant_call(
        "Refactoring.",
        "call_002",
        "write_file",
        json!({
            "path":"tests/StringReverseTest.php",
            "contents":"<?php\nuse PHPUnit\\Framework\\Attributes\\DataProvider;\nuse PHPUnit\\Framework\\TestCase;\n\nclass StringReverseTest extends TestCase {\n    public static function cases(): array {\n        return [\n            'empty'   => ['', ''],\n            'ascii'   => ['hello', 'olleh'],\n            'accent'  => ['héllo', 'olléh'],\n        ];\n    }\n\n    #[DataProvider('cases')]\n    public function testReverse(string $in, string $want): void {\n        $this->assertSame($want, str_reverse_unicode($in));\n    }\n}\n"
        }),
        &turns,
    ));
    turns.push(tool(
        "call_002",
        "wrote 524 bytes to tests/StringReverseTest.php",
    ));
    turns.push(assistant_text(
        "Three tests collapsed into one data-driven test using PHPUnit 10's attribute syntax.",
        &turns,
    ));

    assemble(
        "agentperf_tuning_037",
        "php",
        src(
            "github.com/sebastianbergmann/phpunit",
            "BSD-3-Clause",
            "10.5",
        ),
        turns,
    )
}
